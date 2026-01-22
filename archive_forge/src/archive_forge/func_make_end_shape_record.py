import os
import zlib
import time  # noqa
import logging
import numpy as np
def make_end_shape_record(self):
    bits = BitArray()
    bits += '0'
    bits += '0' * 5
    return bits