from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def has_next_box(self):
    if self.has_length:
        return self.fp.tell() + self.remaining_in_box < self.length
    else:
        return True