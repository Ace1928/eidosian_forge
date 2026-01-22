import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
@staticmethod
def semicolon_at_end_of_expression(expression):
    """Parse Python expression and detects whether last token is ';'"""
    sio = _io.StringIO(expression)
    tokens = list(tokenize.generate_tokens(sio.readline))
    for token in reversed(tokens):
        if token[0] in (tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            continue
        if token[0] == tokenize.OP and token[1] == ';':
            return True
        else:
            return False