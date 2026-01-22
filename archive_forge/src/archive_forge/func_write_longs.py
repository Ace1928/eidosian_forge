import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))