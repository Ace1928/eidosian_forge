import re
from .ply import lex
from .ply.lex import TOKEN
def t_pppragma_PPPRAGMA(self, t):
    """pragma"""
    return t