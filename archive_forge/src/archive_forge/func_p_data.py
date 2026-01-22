import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple
def p_data(p):
    """
    data : data NUM_VAL
         | data WORD
         | data STRING
         | data QUOTEDSTRING
         | data BRACKETEDSTRING
         | data SET
         | data TABLE
         | data PARAM
         | data LPAREN
         | data RPAREN
         | data COMMA
         | data ASTERISK
         | NUM_VAL
         | WORD
         | STRING
         | QUOTEDSTRING
         | BRACKETEDSTRING
         | SET
         | TABLE
         | PARAM
         | LPAREN
         | RPAREN
         | COMMA
         | ASTERISK
    """
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[2]
    if single_item:
        p[0] = [tmp]
    else:
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst