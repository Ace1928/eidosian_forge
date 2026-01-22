import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_BUILD_LIST(self, instr):
    if not instr.arg:
        return
    if not self.build_tuple(instr, tuple):
        self.build_tuple_unpack_seq(instr)