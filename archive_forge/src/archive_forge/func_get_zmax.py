import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
from . import config as inductor_config
def get_zmax(self):
    zmax = inductor_config.triton.max_block['Z']
    if self.size_hints and len(self.size_hints) > 2:
        zmax = min(zmax, self.size_hints[2])
    return zmax