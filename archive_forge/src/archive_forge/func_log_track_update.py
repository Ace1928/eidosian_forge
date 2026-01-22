import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def log_track_update(log_track: int) -> bool:
    """count (log_track[0]) up to threshold (log_track[1]), reset count (log_track[0]) and return true when reached"""
    log_track[LOG_TRACK_COUNT] += 1
    if log_track[LOG_TRACK_COUNT] < log_track[LOG_TRACK_THRESHOLD]:
        return False
    log_track[LOG_TRACK_COUNT] = 0
    return True