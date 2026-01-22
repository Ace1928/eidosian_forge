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
def p_items(p):
    """
    items : items NUM_VAL
          | items WORD
          | items STRING
          | items QUOTEDSTRING
          | items COMMA
          | items COLON
          | items LBRACE
          | items RBRACE
          | items LBRACKET
          | items RBRACKET
          | items TR
          | items LPAREN
          | items RPAREN
          | items ASTERISK
          | items EQ
          | items SET
          | items TABLE
          | items PARAM
          | NUM_VAL
          | WORD
          | STRING
          | QUOTEDSTRING
          | COMMA
          | COLON
          | LBRACKET
          | RBRACKET
          | LBRACE
          | RBRACE
          | TR
          | LPAREN
          | RPAREN
          | ASTERISK
          | EQ
          | SET
          | TABLE
          | PARAM
    """
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[2]
    if type(tmp) is str and tmp[0] == '"' and (tmp[-1] == '"') and (len(tmp) > 2) and (not ' ' in tmp):
        tmp = tmp[1:-1]
    if single_item:
        p[0] = [tmp]
    else:
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst