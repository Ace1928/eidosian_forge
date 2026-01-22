import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def replace_load_const(self, nconst, instr, result):
    self.in_consts = True
    load_const = Instr('LOAD_CONST', result, lineno=instr.lineno)
    start = self.index - nconst - 1
    self.block[start:self.index] = (load_const,)
    self.index -= nconst
    if nconst:
        del self.const_stack[-nconst:]
    self.const_stack.append(result)
    self.in_consts = True