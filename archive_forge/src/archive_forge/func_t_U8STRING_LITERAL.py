import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u8string_literal)
def t_U8STRING_LITERAL(self, t):
    return t