import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def max_line(self, *args):
    m = self.SMALL_LINE_INT
    for arg in args:
        if isinstance(arg, (list, tuple)):
            m = max(m, self.max_line(*arg))
        elif isinstance(arg, _MsgPart):
            m = max(m, arg.line)
        elif hasattr(arg, 'offset'):
            m = max(m, self.op_offset_to_line[arg.offset])
    return m