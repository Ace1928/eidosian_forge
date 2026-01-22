import re
from .ply import lex
from .ply.lex import TOKEN
@TOKEN(u32string_literal)
def t_U32STRING_LITERAL(self, t):
    return t