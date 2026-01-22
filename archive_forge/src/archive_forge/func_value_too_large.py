import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def value_too_large(self, name, val):
    if name == 'XBLOCK':
        return val > self.get_xmax()
    if name == 'YBLOCK':
        return val > self.get_ymax()
    if name == 'ZBLOCK':
        return val > self.get_zmax()
    if name == 'RBLOCK':
        return val > self.get_rmax()
    if name == 'num_warps':
        return val > self.get_warpsmax()
    return False