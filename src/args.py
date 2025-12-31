#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import inspect
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple

from loguru import logger

from .utils import hf_get_or_download


@dataclass
class BaseArgs:
    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return copy.deepcopy(self)

    def to_dict(self, exclude=[]):
        return {k: v for k, v in asdict(self).items() if k not in exclude}

    def fill(self, body: Any):
        sig = inspect.signature(body)
        params = {}
        for key, value in sig.parameters.items():
            if key == "kwargs":
                continue

            if hasattr(self, key):
                params[key] = getattr(self, key)
            else:
                params[key] = value.default

        logger.info(f"{body.__name__}: {params}")

        return params


@dataclass
class SFTArgs(BaseArgs):
    # --------- model args ---------
    model: str = "unsloth/Qwen3-1.7B"
    device: int = 0
    max_seq_length: int = 4096
    dtype: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # --------- train/eval data args ---------
    train_datasets: List[Tuple[str, float]] = field(
        default_factory=lambda: [
            # ("common/sft-common", 0.1),
            # ("common/ruozhiba", 0.1),
            ("common/chumor", 1.0),
        ]
    )

    # --------- sft lora args ---------
    rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_rslora: bool = False
    loftq_config: str = None
    use_gradient_checkpointing: str | bool = False
    layers_to_transform: List[int] | None = None
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # --------- training args ---------
    optim: str = "adamw_torch_fused"
    dataset_num_proc: int = 1
    packing: bool = True
    num_train_epochs: int = 50  # 训练轮数
    per_device_train_batch_size: int = 4  # 每个设备的 batch size
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50  # 用 200 个样本热身
    warmup_ratio: float = 0.05
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    train_on_responses_only: bool = False

    # --------- logging & saving args ---------
    max_steps: int = -1
    save_strategy: str = "epoch"
    save_steps: int = -1
    save_total_limit: int = 10
    evaluation_strategy: str = "epoch"
    eval_steps: int = -1
    logging_strategy: str = "steps"
    logging_steps: int = 25
    output_dir: str = "checkpoints/sft"
    report_to: str = "swanlab"
    run_name: str = str(datetime.now().strftime("%Y%m%d-%H:%M"))
    project: str = "unsloth-demo"

    # --------- precision args ---------
    seed: int = 3407
    bf16: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    def __post_init__(self):
        self.model = hf_get_or_download(self.model)
        os.system(f"mv {self.output_dir} {self.output_dir}_old")
