from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
def set_pixel(self, x, y, color):
    try:
        self.pixels[y][x] = color
    except TypeError:
        self.pixels[y][x] = color[0]