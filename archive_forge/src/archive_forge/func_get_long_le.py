import warnings
from collections import namedtuple
def get_long_le(b):
    return b[3] << 24 | b[2] << 16 | b[1] << 8 | b[0]