from io import StringIO
from typing import List, Union
from bytecode.bytecode import (  # noqa
from bytecode.cfg import BasicBlock, ControlFlowGraph  # noqa
from bytecode.concrete import _ConvertBytecodeToConcrete  # noqa
from bytecode.concrete import ConcreteBytecode, ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import (  # noqa
from bytecode.version import __version__
def format_try_begin(instr: TryBegin, labels: dict) -> str:
    if isinstance(instr.target, Label):
        try:
            arg = '<%s>' % labels[instr.target]
        except KeyError:
            arg = '<error: unknown label>'
    else:
        try:
            arg = '<%s>' % labels[id(instr.target)]
        except KeyError:
            arg = '<error: unknown label>'
    line = 'TryBegin %s -> %s [%s]' % (len(try_begins), arg, instr.stack_depth) + (' last_i' if instr.push_lasti else '')
    try_begins.append(instr)
    return line