import sys
from enum import (
from typing import (
def rl_force_redisplay() -> None:
    """
    Causes readline to display the prompt and input text wherever the cursor is and start
    reading input from this location. This is the proper way to restore the input line after
    printing to the screen
    """
    if not sys.stdout.isatty():
        return
    if rl_type == RlType.GNU:
        readline_lib.rl_forced_update_display()
        display_fixed = ctypes.c_int.in_dll(readline_lib, 'rl_display_fixed')
        display_fixed.value = 1
    elif rl_type == RlType.PYREADLINE:
        readline.rl.mode._print_prompt()
        readline.rl.mode._update_line()