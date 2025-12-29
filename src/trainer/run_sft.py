import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
import re
from importlib import import_module

import swanlab
import torch
import tyro
import unsloth
from jinja2 import Template
from loguru import logger
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, FastModel
from unsloth.chat_templates import train_on_responses_only

from datasets import concatenate_datasets

from ..args import SFTArgs
from ..datasets.hfds_ext import TK_HFDataset

args = tyro.cli(SFTArgs)


# 初始化 WandB run（可选，Trainer 也会自动 init）
# swanlab.init(project=args.project, name=args.run_name, config=args.to_dict())

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    dtype=torch.bfloat16,
    local_files_only=True,
    max_seq_length=args.max_seq_length,  # Choose any for long context!
    load_in_4bit=args.load_in_4bit,  # 4 bit quantization to reduce memory
    load_in_8bit=args.load_in_8bit,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning=False,  # [NEW!] We have full finetuning now!
    fast_inference=False,
    trust_remote_code=False,
    device_map="auto",
    use_gradient_checkpointing="unsloth",
)

logger.info(model)
logger.info(tokenizer)


def import_data(args: SFTArgs):
    train_dss, eval_dss = [], []
    for dataset, w in args.train_datasets:
        module = re.sub("/", ".", dataset.strip("/"))
        module = "src.datasets." + module
        module = import_module(module)
        ds = module.Dataset()
        tds = ds.get_train(w)
        eds = ds.get_validation(w)
        train_dss.append(tds)
        eval_dss.append(eds)

    train_dss = concatenate_datasets(train_dss).shuffle(seed=42)
    eval_dss = concatenate_datasets(eval_dss).shuffle(seed=42)
    train_dss = TK_HFDataset(train_dss)
    eval_dss = TK_HFDataset(eval_dss)
    return train_dss, eval_dss


def formatting_func(example):
    # plain = {k: example[k][0:2] for k in example}
    # print(json.dumps(plain, ensure_ascii=False, indent=2))
    # #print(len(plain['prompts']), len(plain['responses']))

    prompts = []
    for p, t, r in zip(example["prompt"], example["thought"], example["chosen"]):
        messages = [
            {"role": "system", "content": "请以弱智吧用户的思考方式，回答用户问题"},
            {"role": "user", "content": p},
            {"role": "assistant", "content": t + "\n\n回复: " + r},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt)
        logger.info(repr(prompt))

    return prompts


if __name__ == "__main__":
    train_ds, eval_ds = import_data(args)

    logger.info(args.to_dict())
    logger.info(f"train_ds: {len(train_ds)}")
    logger.info(f"eval_ds: {len(eval_ds)}")

    peft_args = args.fill(FastLanguageModel.get_peft_model)

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,  # Supports any, but = 0 is optimized
        bias=args.bias,  # Supports any, but = "none" is optimized
        layers_to_transform=args.layers_to_transform,
        use_gradient_checkpointing=args.use_gradient_checkpointing,  # True or "unsloth" for very long context
        max_seq_length=args.max_seq_length,
        use_rslora=args.use_rslora,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    sft_args = args.fill(SFTConfig)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        args=SFTConfig(**sft_args),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>\n\n</think>\n\n",
    )
    trainer.train()

    # merged_hf_path = args.output_dir + '/checkpoint-full'
    # model.save_pretrained_merged(merged_hf_path, tokenizer, save_method="merged_16bit")
