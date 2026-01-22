from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def rgb_to_hsluv(_hx_tuple: RGBColor) -> Triplet:
    return lch_to_hsluv(rgb_to_lch(_hx_tuple))