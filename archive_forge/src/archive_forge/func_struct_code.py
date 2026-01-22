from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def struct_code(self):
    return '<' if self is Endianness.little else '>'