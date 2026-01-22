import base64
import mimetypes
import os
from html import escape
from typing import Any, Callable, Dict, Iterable, Match, Optional, Tuple
import bs4
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from nbconvert.filters.strings import add_anchor
def parse_latex_environment(self, m: Match[str], state: Any) -> Tuple[str, str, str]:
    """Parse a latex environment."""
    name, text = (m.group(1), m.group(2))
    return ('latex_environment', name, text)