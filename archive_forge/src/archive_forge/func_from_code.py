import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
@staticmethod
def from_code(code):
    concrete = _bytecode.ConcreteBytecode.from_code(code)
    return concrete.to_bytecode()