import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def getheaderAnim(self, im):
    """Get animation header. To replace PILs getheader()[0]"""
    bb = b'GIF89a'
    bb += intToBin(im.size[0])
    bb += intToBin(im.size[1])
    bb += b'\x87\x00\x00'
    return bb