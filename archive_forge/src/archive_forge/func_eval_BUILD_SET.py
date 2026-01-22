import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_BUILD_SET(self, instr):
    if not instr.arg:
        return
    self.build_tuple(instr, frozenset)