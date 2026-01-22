import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_LOAD_CONST(self, instr):
    self.in_consts = True
    value = instr.arg
    self.const_stack.append(value)
    self.in_consts = True