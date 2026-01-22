import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def fix_shape(shape, dim_order):
    if dim_order is DimOrder.PRESUMED_CONTIGUOUS:
        return shape
    if dim_order is DimOrder.CHANNELS_LAST:
        return tuple([shape[0]] + list(shape[2:]) + [shape[1]])
    if dim_order is DimOrder.SCALAR_OR_VECTOR:
        assert len(shape) == 0 or len(shape) == 1
        return shape
    if dim_order is DimOrder.UNKNOWN_CONSTANT:
        return shape
    raise Exception(f'Bad dim_order: {dim_order!r}.')