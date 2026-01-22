import platform
import re
from colorsys import rgb_to_hls
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple
from ._palettes import EIGHT_BIT_PALETTE, STANDARD_PALETTE, WINDOWS_PALETTE
from .color_triplet import ColorTriplet
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME
@lru_cache(maxsize=1024)
def get_ansi_codes(self, foreground: bool=True) -> Tuple[str, ...]:
    """Get the ANSI escape codes for this color."""
    _type = self.type
    if _type == ColorType.DEFAULT:
        return ('39' if foreground else '49',)
    elif _type == ColorType.WINDOWS:
        number = self.number
        assert number is not None
        fore, back = (30, 40) if number < 8 else (82, 92)
        return (str(fore + number if foreground else back + number),)
    elif _type == ColorType.STANDARD:
        number = self.number
        assert number is not None
        fore, back = (30, 40) if number < 8 else (82, 92)
        return (str(fore + number if foreground else back + number),)
    elif _type == ColorType.EIGHT_BIT:
        assert self.number is not None
        return ('38' if foreground else '48', '5', str(self.number))
    else:
        assert self.triplet is not None
        red, green, blue = self.triplet
        return ('38' if foreground else '48', '2', str(red), str(green), str(blue))