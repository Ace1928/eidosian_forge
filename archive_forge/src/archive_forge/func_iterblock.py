import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def iterblock(self, block):
    self.block = block
    self.index = 0
    while self.index < len(block):
        instr = self.block[self.index]
        self.index += 1
        yield instr