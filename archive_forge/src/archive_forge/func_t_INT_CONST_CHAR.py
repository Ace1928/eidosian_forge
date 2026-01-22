import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(multicharacter_constant)
def t_INT_CONST_CHAR(self, t):
    return t