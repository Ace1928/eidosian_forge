from __future__ import absolute_import, division, print_function
import sys
import numpy as np
from ..audio.signal import smooth as smooth_signal
from ..processors import BufferProcessor, OnlineProcessor
@property
def max_interval(self):
    """Maximum beat interval [frames]."""
    return self.histogram_processor.max_interval