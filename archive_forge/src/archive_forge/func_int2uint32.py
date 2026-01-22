import os
import zlib
import time  # noqa
import logging
import numpy as np
def int2uint32(i):
    return int(i).to_bytes(4, 'little')