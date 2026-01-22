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
def p_arg(p):
    """
    arg : arg COMMA NUM_VAL
         | arg COMMA WORD
         | arg COMMA STRING
         | arg COMMA QUOTEDSTRING
         | arg COMMA SET
         | arg COMMA TABLE
         | arg COMMA PARAM
         | NUM_VAL
         | WORD
         | STRING
         | QUOTEDSTRING
         | SET
         | TABLE
         | PARAM
    """
    single_item = len(p) == 2
    if single_item:
        tmp = p[1]
    else:
        tmp = p[3]
    if type(tmp) is str and tmp[0] == '"' and (tmp[-1] == '"') and (len(tmp) > 2) and (not ' ' in tmp):
        tmp = tmp[1:-1]
    if single_item:
        p[0] = [tmp]
    else:
        tmp_lst = p[1]
        tmp_lst.append(tmp)
        p[0] = tmp_lst