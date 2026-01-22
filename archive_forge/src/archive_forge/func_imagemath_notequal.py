from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_notequal(self, other):
    return self.apply('ne', self, other, mode='I')