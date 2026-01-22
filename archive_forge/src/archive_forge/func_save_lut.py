from __future__ import annotations
import re
from . import Image, _imagingmorph
def save_lut(self, filename):
    """Save an operator to an mrl file"""
    if self.lut is None:
        msg = 'No operator loaded'
        raise Exception(msg)
    with open(filename, 'wb') as f:
        f.write(self.lut)