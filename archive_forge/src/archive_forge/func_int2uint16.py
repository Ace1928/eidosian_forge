import os
import zlib
import time  # noqa
import logging
import numpy as np
def int2uint16(i):
    return int(i).to_bytes(2, 'little')