from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
def set_accent_color(accent_color: Optional[str]) -> None:
    """Set an accent color to use in help messages. Takes any color supported by `rich`,
    see `python -m rich.color`. Experimental."""
    THEME.border = Style(color=accent_color, dim=True)
    THEME.description = Style(color=accent_color, bold=True)
    THEME.invocation = Style()
    THEME.metavar = Style(color=accent_color, bold=True)
    THEME.metavar_fixed = Style(color='red', bold=True)
    THEME.helptext = Style(dim=True)
    THEME.helptext_required = Style(color='bright_red', bold=True)
    THEME.helptext_default = Style(color='cyan' if accent_color != 'cyan' else 'magenta')