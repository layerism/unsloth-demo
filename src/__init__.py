import random
import sys
from pathlib import Path

import diskcache as dc
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
random.seed(42)
np.random.seed(42)

cache_dir = Path(__file__).parent.parent.resolve() / ".cache"
cache = dc.Cache(cache_dir)


logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<blue>{file}</blue> | <magenta>{function}</magenta>:<yellow>{line}</yellow> "
    "<cyan>{message}</cyan>",
    enqueue=True,
)
