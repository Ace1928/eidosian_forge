from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
def render_default(self, default: DefaultType) -> Text:
    """Render the default as (y) or (n) rather than True/False."""
    yes, no = self.choices
    return Text(f'({yes})' if default else f'({no})', style='prompt.default')