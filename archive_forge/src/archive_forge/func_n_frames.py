from __future__ import annotations
import os
import re
from . import Image, ImageFile, ImagePalette
@property
def n_frames(self):
    return self.info[FRAMES]