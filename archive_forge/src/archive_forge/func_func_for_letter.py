import re
from functools import partial
from typing import Any, Callable, Dict, Tuple
from curtsies.formatstring import fmtstr, FmtStr
from curtsies.termformatconstants import (
from ..config import COLOR_LETTERS
from ..lazyre import LazyReCompile
def func_for_letter(letter_color_code: str, default: str='k') -> Callable[..., FmtStr]:
    """Returns FmtStr constructor for a bpython-style color code"""
    if letter_color_code == 'd':
        letter_color_code = default
    elif letter_color_code == 'D':
        letter_color_code = default.upper()
    return partial(fmtstr, fg=CNAMES[letter_color_code.lower()], bold=letter_color_code.isupper())