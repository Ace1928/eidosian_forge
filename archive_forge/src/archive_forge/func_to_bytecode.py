import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr
from _pydevd_frame_eval.vendored.bytecode.flags import CompilerFlags
from _pydevd_frame_eval.vendored.bytecode.instr import Label, SetLineno, Instr
def to_bytecode(self):
    """Convert to Bytecode."""
    used_blocks = set()
    for block in self:
        target_block = block.get_jump()
        if target_block is not None:
            used_blocks.add(id(target_block))
    labels = {}
    jumps = []
    instructions = []
    for block in self:
        if id(block) in used_blocks:
            new_label = Label()
            labels[id(block)] = new_label
            instructions.append(new_label)
        for instr in block:
            if isinstance(instr, Instr):
                instr = instr.copy()
                if isinstance(instr.arg, BasicBlock):
                    jumps.append(instr)
            instructions.append(instr)
    for instr in jumps:
        instr.arg = labels[id(instr.arg)]
    bytecode = _bytecode.Bytecode()
    bytecode._copy_attr_from(self)
    bytecode.argnames = list(self.argnames)
    bytecode[:] = instructions
    return bytecode