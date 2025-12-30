import gzip
import json
import re
import shutil
from pathlib import Path

from loguru import logger

from datasets import load_dataset

from ...utils import hf_get_or_download
from ..hfds_ext import HFDSLoader


def preprocess(data_dir):
    data_path = Path(data_dir)
    unzip_dir = data_path / "unzip"
    unzip_dir.mkdir(parents=True, exist_ok=True)

    for file_path in data_path.glob("*.gz"):
        output_file_name = file_path.stem  # This removes .gz
        output_path = unzip_dir / output_file_name

        if output_path.exists():
            logger.info(f"File {output_path} already exists, skipping unzip.")
            continue

        logger.info(f"Unzipping {file_path} to {output_path}")
        with gzip.open(file_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


class Dataset(HFDSLoader):
    def __init__(
        self,
        cache_dir="/tmp/lccc",
        download_mode="reuse_cache_if_exists",
        **kwargs,
    ) -> None:
        """
        初始化数据集加载器

        Args:
            cache_dir: 指定缓存目录，默认使用 huggingface 默认缓存目录
            download_mode: 下载模式，默认为 "reuse_cache_if_exists" 优先使用缓存
        """
        data_dir = hf_get_or_download("silver/lccc", repo_type="dataset")
        preprocess(data_dir)

        self.dataset = load_dataset(
            path="text",
            data_files={
                "train": f"{data_dir}/unzip/lccc_base_train.jsonl",
                "test": f"{data_dir}/unzip/lccc_base_test.jsonl",
                "validation": f"{data_dir}/unzip/lccc_base_valid.jsonl",
            },
            cache_dir=cache_dir,
            download_mode=download_mode,  # "reuse_cache_if_exists" 或 "force_redownload"
            trust_remote_code=True,
        )

        self.train = self.dataset["train"]
        self.validation = self.dataset["validation"]
        self.test = self.dataset["test"]
        self.all = self.dataset["train"]

    def process_data(self, sample: dict, phase: str = "train") -> dict:
        chat = sample["text"]
        chat = json.loads(chat)
        chat = list(map(lambda x: re.sub(r" ", "", x), chat))

        context, prompt, response = chat[0:-2], chat[-2], chat[-1]

        if context:
            for i in range(len(context)):
                if i % 2 == 0:
                    context[-i - 1] = {"role": "assistant", "content": context[-i - 1]}
                else:
                    context[-i - 1] = {"role": "user", "content": context[-i - 1]}

        data = {
            "context": str(context),
            "prompt": prompt,
            "thought": "",
            "chosen": response,
            "rejected": "",
            "from": "lccc",
        }

        return data


if __name__ == "__main__":
    dataset = Dataset().get_train()
    dataset[0:2].pprint()
