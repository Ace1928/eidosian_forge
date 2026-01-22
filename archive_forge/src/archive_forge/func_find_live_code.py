import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def find_live_code(start):
    for i in range(start, len(instructions)):
        if i in live_code:
            return
        live_code.add(i)
        inst = instructions[i]
        if inst.exn_tab_entry:
            find_live_code(indexof[inst.exn_tab_entry.target])
        if inst.opcode in JUMP_OPCODES:
            find_live_code(indexof[inst.target])
        if inst.opcode in TERMINAL_OPCODES:
            return