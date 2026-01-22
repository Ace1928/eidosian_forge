import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def min_line(self, *args):
    m = self.BIG_LINE_INT
    for arg in args:
        if isinstance(arg, (list, tuple)):
            m = min(m, self.min_line(*arg))
        elif isinstance(arg, _MsgPart):
            m = min(m, arg.line)
        elif hasattr(arg, 'offset'):
            m = min(m, self.op_offset_to_line[arg.offset])
    return m