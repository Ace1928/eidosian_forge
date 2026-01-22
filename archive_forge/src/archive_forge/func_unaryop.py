import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def unaryop(self, op, instr):
    try:
        value = self.const_stack[-1]
        result = op(value)
    except IndexError:
        return
    if not self.check_result(result):
        return
    self.replace_load_const(1, instr, result)