import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def collect_return_info(co, use_func_first_line=False):
    if not hasattr(co, 'co_lines') and (not hasattr(co, 'co_lnotab')):
        return []
    if use_func_first_line:
        firstlineno = co.co_firstlineno
    else:
        firstlineno = 0
    lst = []
    op_offset_to_line = dict(dis.findlinestarts(co))
    for instruction in iter_instructions(co):
        curr_op_name = instruction.opname
        if curr_op_name == 'RETURN_VALUE':
            lst.append(ReturnInfo(_get_line(op_offset_to_line, instruction.offset, firstlineno, search=True)))
    return lst