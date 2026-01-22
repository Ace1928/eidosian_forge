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
def get_truecolor(self, theme: Optional['TerminalTheme']=None, foreground: bool=True) -> ColorTriplet:
    """Get an equivalent color triplet for this color.

        Args:
            theme (TerminalTheme, optional): Optional terminal theme, or None to use default. Defaults to None.
            foreground (bool, optional): True for a foreground color, or False for background. Defaults to True.

        Returns:
            ColorTriplet: A color triplet containing RGB components.
        """
    if theme is None:
        theme = DEFAULT_TERMINAL_THEME
    if self.type == ColorType.TRUECOLOR:
        assert self.triplet is not None
        return self.triplet
    elif self.type == ColorType.EIGHT_BIT:
        assert self.number is not None
        return EIGHT_BIT_PALETTE[self.number]
    elif self.type == ColorType.STANDARD:
        assert self.number is not None
        return theme.ansi_colors[self.number]
    elif self.type == ColorType.WINDOWS:
        assert self.number is not None
        return WINDOWS_PALETTE[self.number]
    else:
        assert self.number is None
        return theme.foreground_color if foreground else theme.background_color