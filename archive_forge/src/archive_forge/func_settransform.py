from __future__ import annotations
from . import Image, ImageColor, ImageDraw, ImageFont, ImagePath
def settransform(self, offset):
    """Sets a transformation offset."""
    xoffset, yoffset = offset
    self.transform = (1, 0, xoffset, 0, 1, yoffset)