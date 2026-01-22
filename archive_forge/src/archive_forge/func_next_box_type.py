from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def next_box_type(self):
    if self.remaining_in_box > 0:
        self.fp.seek(self.remaining_in_box, os.SEEK_CUR)
    self.remaining_in_box = -1
    lbox, tbox = self.read_fields('>I4s')
    if lbox == 1:
        lbox = self.read_fields('>Q')[0]
        hlen = 16
    else:
        hlen = 8
    if lbox < hlen or not self._can_read(lbox - hlen):
        msg = 'Invalid header length'
        raise SyntaxError(msg)
    self.remaining_in_box = lbox - hlen
    return tbox