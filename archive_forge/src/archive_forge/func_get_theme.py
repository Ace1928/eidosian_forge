import os.path
import platform
import re
import sys
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
from pip._vendor.pygments.lexer import Lexer
from pip._vendor.pygments.lexers import get_lexer_by_name, guess_lexer_for_filename
from pip._vendor.pygments.style import Style as PygmentsStyle
from pip._vendor.pygments.styles import get_style_by_name
from pip._vendor.pygments.token import (
from pip._vendor.pygments.util import ClassNotFound
from pip._vendor.rich.containers import Lines
from pip._vendor.rich.padding import Padding, PaddingDimensions
from ._loop import loop_first
from .cells import cell_len
from .color import Color, blend_rgb
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment, Segments
from .style import Style, StyleType
from .text import Text
@classmethod
def get_theme(cls, name: Union[str, SyntaxTheme]) -> SyntaxTheme:
    """Get a syntax theme instance."""
    if isinstance(name, SyntaxTheme):
        return name
    theme: SyntaxTheme
    if name in RICH_SYNTAX_THEMES:
        theme = ANSISyntaxTheme(RICH_SYNTAX_THEMES[name])
    else:
        theme = PygmentsSyntaxTheme(name)
    return theme