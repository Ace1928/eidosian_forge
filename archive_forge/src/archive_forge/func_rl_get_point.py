import sys
from enum import (
from typing import (
def rl_get_point() -> int:
    """
    Returns the offset of the current cursor position in rl_line_buffer
    """
    if rl_type == RlType.GNU:
        return ctypes.c_int.in_dll(readline_lib, 'rl_point').value
    elif rl_type == RlType.PYREADLINE:
        return int(readline.rl.mode.l_buffer.point)
    else:
        return 0