import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import (
import bytecode as _bytecode
from bytecode.concrete import ConcreteInstr
from bytecode.flags import CompilerFlags
from bytecode.instr import UNSET, Instr, Label, SetLineno, TryBegin, TryEnd
def get_dead_blocks(self) -> List[BasicBlock]:
    if not self:
        return []
    seen_block_ids = set()
    stack = [self[0]]
    while stack:
        block = stack.pop()
        if id(block) in seen_block_ids:
            continue
        seen_block_ids.add(id(block))
        for i in block:
            if isinstance(i, Instr) and isinstance(i.arg, BasicBlock):
                stack.append(i.arg)
            elif isinstance(i, TryBegin):
                assert isinstance(i.target, BasicBlock)
                stack.append(i.target)
    return [b for b in self if id(b) not in seen_block_ids]