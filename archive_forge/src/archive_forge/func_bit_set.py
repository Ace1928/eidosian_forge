import math
import os
from struct import pack, unpack, calcsize
from typing import BinaryIO, Dict, Iterable, List, Optional, Tuple, Union, cast
def bit_set(bit_pos: int, value: int) -> bool:
    return bool(value >> bit_pos & 1)