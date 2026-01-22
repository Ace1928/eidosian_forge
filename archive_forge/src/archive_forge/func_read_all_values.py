import json
import mmap
import os
import struct
from typing import List
def read_all_values(self):
    """Yield (key, value, timestamp). No locking is performed."""
    for k, v, ts, _ in self._read_all_values():
        yield (k, v, ts)