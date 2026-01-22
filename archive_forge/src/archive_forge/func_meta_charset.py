import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
@contextmanager
def meta_charset(tmp_charset):
    global _charset
    old = _charset
    _charset = tmp_charset
    yield
    _charset = old