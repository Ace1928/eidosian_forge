from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import (
from _pydevd_frame_eval.vendored.bytecode.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import (
from _pydevd_frame_eval.vendored.bytecode.cfg import BasicBlock, ControlFlowGraph  # noqa
import sys
def format_instr(instr, labels=None):
    text = instr.name
    arg = instr._arg
    if arg is not UNSET:
        if isinstance(arg, Label):
            try:
                arg = '<%s>' % labels[arg]
            except KeyError:
                arg = '<error: unknown label>'
        elif isinstance(arg, BasicBlock):
            try:
                arg = '<%s>' % labels[id(arg)]
            except KeyError:
                arg = '<error: unknown block>'
        else:
            arg = repr(arg)
        text = '%s %s' % (text, arg)
    return text