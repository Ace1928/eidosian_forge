import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def eval_BUILD_TUPLE(self, instr):
    if not instr.arg:
        return
    if instr.arg <= len(self.const_stack):
        self.replace_container_of_consts(instr, tuple)
    else:
        self.build_tuple_unpack_seq(instr)