import multiprocessing
import sys
import torch
from .reductions import init_reductions
from multiprocessing import *  # noqa: F403
from .spawn import (
def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy