import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def compute_exception_table(instructions: List[Instruction]) -> List[ExceptionTableEntry]:
    """Compute exception table in list format from instructions with exn_tab_entries"""
    exn_dict: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
    indexof = get_indexof(instructions)
    for inst in instructions:
        if inst.exn_tab_entry:
            start = _get_instruction_front(instructions, indexof[inst.exn_tab_entry.start]).offset
            end = cast(int, inst.exn_tab_entry.end.offset) + instruction_size(inst.exn_tab_entry.end) - 2
            target = _get_instruction_front(instructions, indexof[inst.exn_tab_entry.target]).offset
            key = (start, end)
            val = (target, inst.exn_tab_entry.depth, inst.exn_tab_entry.lasti)
            if key in exn_dict:
                assert exn_dict[key] == val
            exn_dict[key] = val
    keys_sorted = sorted(exn_dict.keys(), key=lambda t: (t[0], -t[1]))
    nexti = 0
    key_stack: List[Tuple[int, int]] = []
    exn_tab: List[ExceptionTableEntry] = []

    def pop():
        """
        Pop the key_stack and append an exception table entry if possible.
        """
        nonlocal nexti
        if key_stack:
            key = key_stack.pop()
            if nexti <= key[1]:
                exn_tab.append(ExceptionTableEntry(max(key[0], nexti), key[1], *exn_dict[key]))
                nexti = key[1] + 2
    for key in keys_sorted:
        while key_stack and key_stack[-1][1] < key[0]:
            pop()
        if key_stack:
            assert key_stack[-1][0] <= key[0] <= key[1] <= key_stack[-1][1]
            left = max(nexti, key_stack[-1][0])
            if left < key[0]:
                exn_tab.append(ExceptionTableEntry(left, key[0] - 2, *exn_dict[key_stack[-1]]))
            nexti = key[0]
        key_stack.append(key)
    while key_stack:
        pop()
    check_exception_table(exn_tab)
    return exn_tab