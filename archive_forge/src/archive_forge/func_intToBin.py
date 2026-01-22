import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
def intToBin(i):
    return i.to_bytes(2, byteorder='little')