import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_UNARY_NOT(self, instr):
    next_instr = self.get_next_instr('POP_JUMP_IF_FALSE')
    if next_instr is None:
        return None
    instr.set('POP_JUMP_IF_TRUE', next_instr.arg)
    del self.block[self.index]