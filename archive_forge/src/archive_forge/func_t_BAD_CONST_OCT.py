import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(bad_octal_constant)
def t_BAD_CONST_OCT(self, t):
    msg = 'Invalid octal constant'
    self._error(msg, t)