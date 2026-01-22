from functools import partial
from importlib import import_module
from typing import Any, Dict, Optional, Type, Union
from pygments import highlight
from pygments.filters import ErrorToken
from pygments.formatter import Formatter
from pygments.formatters import HtmlFormatter, LatexFormatter
from pygments.lexer import Lexer
from pygments.lexers import (CLexer, PythonConsoleLexer, PythonLexer, RstLexer, TextLexer,
from pygments.style import Style
from pygments.styles import get_style_by_name
from pygments.util import ClassNotFound
from sphinx.locale import __
from sphinx.pygments_styles import NoneStyle, SphinxStyle
from sphinx.util import logging, texescape
Highlight code blocks using Pygments.