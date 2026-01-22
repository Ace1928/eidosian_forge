from collections import namedtuple
import dis
from functools import partial
import itertools
import os.path
import sys
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import Instr, Label
from _pydev_bundle import pydev_log
from _pydevd_frame_eval.pydevd_frame_tracing import _pydev_stop_at_break, _pydev_needs_stop_at_break
def get_instructions_to_add(stop_at_line, _pydev_stop_at_break=_pydev_stop_at_break, _pydev_needs_stop_at_break=_pydev_needs_stop_at_break):
    """
    This is the bytecode for something as:

        if _pydev_needs_stop_at_break():
            _pydev_stop_at_break()

    but with some special handling for lines.
    """
    spurious_line = stop_at_line - 1
    if spurious_line <= 0:
        spurious_line = stop_at_line + 1
    label = Label()
    return [Instr('LOAD_CONST', _pydev_needs_stop_at_break, lineno=stop_at_line), Instr('LOAD_CONST', stop_at_line, lineno=stop_at_line), Instr('CALL_FUNCTION', 1, lineno=stop_at_line), Instr('POP_JUMP_IF_FALSE', label, lineno=stop_at_line), Instr('LOAD_CONST', _pydev_stop_at_break, lineno=spurious_line), Instr('LOAD_CONST', stop_at_line, lineno=spurious_line), Instr('CALL_FUNCTION', 1, lineno=spurious_line), Instr('POP_TOP', lineno=spurious_line), Instr('NOP', lineno=stop_at_line), label]