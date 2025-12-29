import os
import subprocess
from pathlib import Path

from loguru import logger


def hf_get_or_download(
    repo_id: str,
    repo_type: str = "model",
    force_download: bool = False,
    rename: str = None,
) -> None:
    from huggingface_hub import HfFolder, login, snapshot_download
    from transformers import TRANSFORMERS_CACHE

    # download model to ~/.cache/huggingface/hub
    if rename is None:
        rename = repo_id.split("/")[-1]

    env = "MODEL-{}".format(rename.upper())
    logger.info("{} --> Using default HF cache directory".format(env))

    # token = HfFolder.get_token()
    # if not token:
    #     login(token=os.getenv("HF_TOKEN"))

    model_cache = "/{}s--{}/snapshots".format(repo_type, repo_id.replace("/", "--"))
    repo_default_cache = TRANSFORMERS_CACHE + model_cache
    repo_default_cache = os.path.expanduser(repo_default_cache)

    if force_download:
        subprocess.run(["rm", "-rf", repo_default_cache])

    local_files_only = False
    if Path(repo_default_cache).exists():
        local_files_only = True

    local_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type, local_files_only=local_files_only)

    logger.info(f"{repo_type.upper()} Repo Path: {local_dir}")

    return local_dir
