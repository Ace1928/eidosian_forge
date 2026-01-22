from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def pieslice(self, xy, start, end, fill=None, outline=None, width=1):
    """Draw a pieslice."""
    ink, fill = self._getink(outline, fill)
    if fill is not None:
        self.draw.draw_pieslice(xy, start, end, fill, 1)
    if ink is not None and ink != fill and (width != 0):
        self.draw.draw_pieslice(xy, start, end, ink, 0, width)