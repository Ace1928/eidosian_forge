import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(hex_floating_constant)
def t_HEX_FLOAT_CONST(self, t):
    return t