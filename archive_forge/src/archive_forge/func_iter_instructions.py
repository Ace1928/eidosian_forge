import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
def iter_instructions(co):
    if sys.version_info[0] < 3:
        iter_in = _iter_as_bytecode_as_instructions_py2(co)
    else:
        iter_in = dis.Bytecode(co)
    iter_in = list(iter_in)
    bytecode_to_instruction = {}
    for instruction in iter_in:
        bytecode_to_instruction[instruction.offset] = instruction
    if iter_in:
        for instruction in iter_in:
            yield instruction