import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def use_nchw(self):
    if self.dim_order is DimOrder.PRESUMED_CONTIGUOUS:
        return True
    if self.dim_order is DimOrder.CHANNELS_LAST:
        return False
    raise Exception('Unknown dim order')