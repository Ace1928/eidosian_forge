import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
@property
def aux_arrays(self):
    """Shared aux states."""
    return self.execgrp.aux_arrays