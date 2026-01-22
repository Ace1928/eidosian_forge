import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_UNARY_NEGATIVE(self, instr):
    return self.unaryop(operator.neg, instr)