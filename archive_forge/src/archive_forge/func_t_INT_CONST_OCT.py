import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(octal_constant)
def t_INT_CONST_OCT(self, t):
    return t