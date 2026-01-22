import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def download_progress_hook(count: int, block_size: int, total_size: int) -> None:
    if pbar.total is None:
        pbar.total = total_size
        pbar.reset()
    downloaded_size = count * block_size
    pbar.update(downloaded_size - pbar.n)
    if pbar.n >= total_size:
        pbar.close()