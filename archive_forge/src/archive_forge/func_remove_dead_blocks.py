import opcode
import operator
import sys
from _pydevd_frame_eval.vendored.bytecode import Instr, Bytecode, ControlFlowGraph, BasicBlock, Compare
def remove_dead_blocks(self):
    used_blocks = {id(self.code[0])}
    for block in self.code:
        if block.next_block is not None:
            used_blocks.add(id(block.next_block))
        for instr in block:
            if isinstance(instr, Instr) and isinstance(instr.arg, BasicBlock):
                used_blocks.add(id(instr.arg))
    block_index = 0
    while block_index < len(self.code):
        block = self.code[block_index]
        if id(block) not in used_blocks:
            del self.code[block_index]
        else:
            block_index += 1