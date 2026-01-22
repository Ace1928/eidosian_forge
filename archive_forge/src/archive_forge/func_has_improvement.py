import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
@staticmethod
def has_improvement(baseline, test):
    threshold = 0.001
    return test is not None and test < baseline * (1 - threshold)