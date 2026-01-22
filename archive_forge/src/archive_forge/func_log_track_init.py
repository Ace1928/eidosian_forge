import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def log_track_init(log_freq: int) -> List[int]:
    """create tracking structure used by log_track_update"""
    l = [0] * 2
    l[LOG_TRACK_THRESHOLD] = log_freq
    return l