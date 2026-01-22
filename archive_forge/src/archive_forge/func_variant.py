import os
import sys
from enum import Enum, _simple_enum
@property
def variant(self):
    if not self.int & 32768 << 48:
        return RESERVED_NCS
    elif not self.int & 16384 << 48:
        return RFC_4122
    elif not self.int & 8192 << 48:
        return RESERVED_MICROSOFT
    else:
        return RESERVED_FUTURE