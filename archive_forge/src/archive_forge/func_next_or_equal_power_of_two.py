import os
import math
import sys
from typing import Optional, Union, Callable
import pyglet
from pyglet.customtypes import Buffer
def next_or_equal_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << math.ceil(math.log2(x))