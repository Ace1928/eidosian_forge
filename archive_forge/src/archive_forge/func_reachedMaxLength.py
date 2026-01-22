import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def reachedMaxLength(self, out):
    assert out.pos == out.size
    return self.allocated == self.max_length