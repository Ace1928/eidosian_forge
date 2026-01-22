from __future__ import annotations
import array
from . import GimpGradientFile, GimpPaletteFile, ImageColor, PaletteFile
@palette.setter
def palette(self, palette):
    self._colors = None
    self._palette = palette