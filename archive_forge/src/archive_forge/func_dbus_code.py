from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
def dbus_code(self):
    return b'l' if self is Endianness.little else b'B'