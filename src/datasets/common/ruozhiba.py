import gzip
import json
import re
import shutil
from pathlib import Path

from loguru import logger

from datasets import load_dataset

from ...utils import hf_get_or_download
from ..hfds_ext import HFDSLoader


class Dataset(HFDSLoader):
    def __init__(
        self,
        cache_dir="/tmp/ruozhiba",
        download_mode="reuse_cache_if_exists",
        **kwargs,
    ) -> None:
        """
        初始化数据集加载器

        Args:
            cache_dir: 指定缓存目录，默认使用 huggingface 默认缓存目录
            download_mode: 下载模式，默认为 "reuse_cache_if_exists" 优先使用缓存
        """
        data_dir = hf_get_or_download("LooksJuicy/ruozhiba-punchline", repo_type="dataset")

        self.dataset = load_dataset(
            path="json",
            data_files={
                "train": f"{data_dir}/joke_result_sample.json",
            },
            cache_dir=cache_dir,
            download_mode=download_mode,  # "reuse_cache_if_exists" 或 "force_redownload"
            trust_remote_code=True,
        )

        data = self.dataset["train"]
        a, b, n = int(len(data) * 0.9), int(len(data) * 0.95), len(data)
        self.train = data.select(range(0, a))
        self.validation = data.select(range(a, b))
        self.test = data.select(range(b, n))
        self.all = data

    def process_data(self, sample: dict, phase: str = "train") -> dict:
        data = {
            "context": "",
            "prompt": sample["instruction"],
            "thought": sample["thought"],
            "chosen": sample["output"],
            "rejected": "",
            "from": "ruozhiba",
        }

        return data


if __name__ == "__main__":
    dataset = Dataset().get_train()
    dataset[0:3].pprint()
