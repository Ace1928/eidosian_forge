import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u16char_const)
def t_U16CHAR_CONST(self, t):
    return t