import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
def readints(fd, n):
    a = np.frombuffer(fd.read(int(n * 8)), dtype=np.int64, count=n)
    if not np.little_endian:
        a = a.byteswap()
    return a