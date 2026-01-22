import os
import zlib
import time  # noqa
import logging
import numpy as np
def bits2int(bb, n=8):
    value = ''
    for i in range(len(bb)):
        b = bb[i:i + 1]
        tmp = bin(ord(b))[2:]
        value = tmp.rjust(8, '0') + value
    return int(value[:n], 2)