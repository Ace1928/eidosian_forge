import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def str2long(s):
    warnings.warn('str2long() has been replaced by bytes_to_long()')
    return bytes_to_long(s)