import os
import random
import time
from ._compat import long, binary_type
def random_between(self, first, last):
    size = last - first + 1
    if size > long(4294967296):
        raise ValueError('too big')
    if size > 65536:
        rand = self.random_32
        max = long(4294967295)
    elif size > 256:
        rand = self.random_16
        max = 65535
    else:
        rand = self.random_8
        max = 255
    return first + size * rand() // (max + 1)