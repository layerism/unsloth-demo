import json
import os
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger

from datasets import Dataset as HFDataset
from datasets import load_dataset


class DictSample(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pprint(self):
        print(json.dumps(self, indent=2, ensure_ascii=False))

    def tokenized_by(self, tokenizer):
        # 检查是否有 messages 字段
        if "messages" in self:
            data = self["messages"]
        else:
            data = self

        fmt_text = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(fmt_text, add_special_tokens=True, max_length=2048)
        tokens = []
        for token_id in token_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            tokens.append(token_text)

        print([fmt_text], tokens)

        for id, (token, token_id) in enumerate(zip(tokens, token_ids)):
            print(f"{id:4d} | {token_id:8d} | {repr(token):20s}")

        return tokens, token_ids


class TK_HFDataset(HFDataset):
    """Wrapper for Hugging Face Dataset, automatically converts to DictSample"""

    def __init__(self, hf_dataset):
        super().__init__(
            arrow_table=hf_dataset.data,
            info=hf_dataset.info,
            split=hf_dataset.split,
            indices_table=getattr(hf_dataset, "indices", None),
            fingerprint=getattr(hf_dataset, "fingerprint", None),
        )

    def __getitem__(self, key: Union[int, slice, str, List[int], List[str]]) -> Any:
        """
        Get items from the dataset.
        - Using integer index (int): returns DictSample.
        - Using slice or list of integer indices (List[int]): returns list of DictSample.
        - Using column name (str): returns original column data (usually a list).
        - Using list of column names (List[str]): returns a new TK_HFDataset (containing selected columns).
        """
        item = super().__getitem__(key)

        if isinstance(item, dict):
            item = DictSample(item)
        elif isinstance(item, list) and all(isinstance(x, dict) for x in item):
            item = [DictSample(x) for x in item]
        elif isinstance(item, HFDataset):
            item = TK_HFDataset(item)
        else:
            pass

        return item

    def __iter__(self):
        for data in super().__iter__():
            yield DictSample(data)

    def sample(self):
        indice = random.sample(range(len(self)), 1)[0]
        return self[indice]


class HFDSLoader(ABC):
    @abstractmethod
    def process_data(self, sample: dict) -> dict:
        pass

    def split(self, phase: str, weight: float = 1.0, redo: bool = True, num_proc: int = 1, seed: int = 42):
        assert hasattr(self, "train") and hasattr(self, "test")

        if phase == "train":
            split_ds = self.train
        elif phase == "validation":
            split_ds = self.validation
        elif phase == "test":
            split_ds = self.test
        elif phase == "all":
            split_ds = self.all
        else:
            raise ValueError(f"Invalid phase: {phase}")

        if redo:
            split_ds.cleanup_cache_files()

        if weight < 1.0:
            tot_n = int(len(split_ds) * weight)
            split_ds = split_ds.shuffle(seed=seed).select(range(tot_n))

        split_ds = split_ds.map(
            partial(self.batch_process_data, phase=phase),
            remove_columns=split_ds.column_names,
            num_proc=None,
            load_from_cache_file=True,
            batched=True,
            batch_size=64,
            keep_in_memory=redo,
        )

        return TK_HFDataset(split_ds)

    def load_dataset(self, *args, **kwargs):
        reload = kwargs.get("reload", False)
        cache_dir = kwargs.get("cache_dir", "/tmp")

        if not os.path.exists(cache_dir) or reload:
            ds = load_dataset(*args, **kwargs)
            ds.save_to_disk(cache_dir)
        else:
            # prefer to load from cache, avoid re-downloading
            ds = load_dataset(path=cache_dir, *args, **kwargs)

        return ds

    def process_data(self, sample: dict, **kwargs):
        return sample

    def batch_process_data(self, samples: Dict[str, List], **kwargs):
        keys = samples.keys()
        length = len(next(iter(samples.values())))
        rows = [{k: samples[k][i] for k in keys} for i in range(length)]

        datas = []
        for row in rows:
            data = self.process_data(row, **kwargs)
            datas.append(data)

        df = pd.DataFrame(datas)
        return df.to_dict(orient="list")

    def get_train(self, weight: float = 1.0):
        return self.split("train", weight=weight)

    def get_validation(self, weight: float = 1.0):
        return self.split("validation", weight=weight)

    def get_test(self, weight: float = 1.0):
        return self.split("test", weight=weight)

    def get_all(self, weight: float = 1.0):
        return self.split("all", weight=weight)

    def add_tokens(self, tokenizer, text):
        return tokenizer.encode(text)

    def remove_tokens(self, tokenizer, text):
        return tokenizer.decode(text)

    # @abstractmethod
    # def expect(self, ans: str, pred: str) -> bool:
    #     pass
