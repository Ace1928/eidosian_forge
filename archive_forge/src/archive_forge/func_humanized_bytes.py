from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def humanized_bytes(self):
    used_bytes = self._humanized_size(self.used_bytes)
    acquired_bytes = self._humanized_size(self.acquired_bytes)
    return (used_bytes, acquired_bytes)