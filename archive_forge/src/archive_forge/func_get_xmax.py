import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def get_xmax(self):
    xmax = inductor_config.triton.max_block['X']
    if self.size_hints and len(self.size_hints) > 0:
        xmax = min(xmax, self.size_hints[0])
    return xmax