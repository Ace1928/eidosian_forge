import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def decode_variable_int(value):
    """Decode a list to a variable length integer.

    Does the opposite of encode_variable_int(value)
    """
    for i in range(len(value) - 1):
        value[i] &= ~128
    val = 0
    for i in value:
        val <<= 7
        val |= i
    return val