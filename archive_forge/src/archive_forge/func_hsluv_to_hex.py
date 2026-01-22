from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def hsluv_to_hex(_hx_tuple: Triplet) -> RGBHexColor:
    return rgb_to_hex(hsluv_to_rgb(_hx_tuple))