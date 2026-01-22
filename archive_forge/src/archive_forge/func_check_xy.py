from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
def check_xy(self, xy):
    x, y = xy
    if not (0 <= x < self.xsize and 0 <= y < self.ysize):
        msg = 'pixel location out of range'
        raise ValueError(msg)
    return xy