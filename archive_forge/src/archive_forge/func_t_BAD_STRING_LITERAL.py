import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(bad_string_literal)
def t_BAD_STRING_LITERAL(self, t):
    msg = 'String contains invalid escape code'
    self._error(msg, t)