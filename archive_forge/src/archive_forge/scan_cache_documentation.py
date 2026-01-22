import time
from argparse import Namespace, _SubParsersAction
from typing import Optional
from ..utils import CacheNotFound, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI, tabulate
Contains command to scan the HF cache directory.

Usage:
    huggingface-cli scan-cache
    huggingface-cli scan-cache -v
    huggingface-cli scan-cache -vvv
    huggingface-cli scan-cache --dir ~/.cache/huggingface/hub
