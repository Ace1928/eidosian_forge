import re
from .ply import lex
from .ply.lex import TOKEN
def t_pppragma_STR(self, t):
    """.+"""
    t.type = 'PPPRAGMASTR'
    return t