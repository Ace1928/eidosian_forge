import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_JUMP_IF_FALSE_OR_POP(self, instr):
    self.jump_if_or_pop(instr)