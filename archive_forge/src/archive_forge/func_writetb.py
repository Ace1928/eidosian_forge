import sys
from codeop import CommandCompiler
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from pygments.token import Generic, Token, Keyword, Name, Comment, String
from pygments.token import Error, Literal, Number, Operator, Punctuation
from pygments.token import Whitespace, _TokenType
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name
from curtsies.formatstring import FmtStr
from ..curtsiesfrontend.parse import parse
from ..repl import Interpreter as ReplInterpreter
def writetb(self, lines: Iterable[str]) -> None:
    tbtext = ''.join(lines)
    lexer = get_lexer_by_name('pytb')
    self.format(tbtext, lexer)