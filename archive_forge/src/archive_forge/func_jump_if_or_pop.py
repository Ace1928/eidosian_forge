import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def jump_if_or_pop(self, instr):
    target_block = instr.arg
    try:
        target_instr = target_block[0]
    except IndexError:
        return
    if not target_instr.is_cond_jump():
        self.optimize_jump_to_cond_jump(instr)
        return
    if (target_instr.name in JUMPS_ON_TRUE) == (instr.name in JUMPS_ON_TRUE):
        target2 = target_instr.arg
        instr.name = target_instr.name
        instr.arg = target2
        self.block[self.index - 1] = instr
        self.index -= 1
    else:
        if instr.name in JUMPS_ON_TRUE:
            name = 'POP_JUMP_IF_TRUE'
        else:
            name = 'POP_JUMP_IF_FALSE'
        new_label = self.code.split_block(target_block, 1)
        instr.name = name
        instr.arg = new_label
        self.block[self.index - 1] = instr
        self.index -= 1