import sys
import re
import copy
import time
import os.path
def t_CPP_COMMENT1(t):
    """(/\\*(.|\\n)*?\\*/)"""
    ncr = t.value.count('\n')
    t.lexer.lineno += ncr
    t.type = 'CPP_WS'
    t.value = '\n' * ncr if ncr else ' '
    return t