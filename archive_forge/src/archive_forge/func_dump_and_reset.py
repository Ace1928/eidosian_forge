import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Set, Tuple
import torch
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import (
@classmethod
def dump_and_reset(cls, msg: str) -> None:
    logger.warning('%s %s', msg, str(cls.results))
    cls.reset()