import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def unsigned(to_type, n):
    return signed(f'u{to_type}', n)