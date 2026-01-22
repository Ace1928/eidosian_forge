import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def const_key(obj):
    try:
        return _dumps(obj)
    except ValueError:
        return (type(obj), id(obj))