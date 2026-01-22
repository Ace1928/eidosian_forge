from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
@property
def overlap_factor(self):
    """Overlapping factor of two adjacent frames."""
    return 1.0 - self.hop_size / self.frame_size