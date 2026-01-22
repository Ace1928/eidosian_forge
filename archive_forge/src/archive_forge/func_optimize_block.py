import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def optimize_block(self, block):
    self.const_stack.clear()
    self.in_consts = False
    for instr in self.iterblock(block):
        if not self.in_consts:
            self.const_stack.clear()
        self.in_consts = False
        meth_name = 'eval_%s' % instr.name
        meth = getattr(self, meth_name, None)
        if meth is not None:
            meth(instr)
        elif instr.has_jump():
            self.optimize_jump(instr)