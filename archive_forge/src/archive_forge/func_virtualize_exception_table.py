import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def virtualize_exception_table(exn_tab_bytes: bytes, instructions: List[Instruction]):
    """Replace exception table entries with pointers to make editing easier"""
    exn_tab = parse_exception_table(exn_tab_bytes)
    offset_to_inst = {cast(int, inst.offset): inst for inst in instructions}
    offsets = sorted(offset_to_inst.keys())
    end_offset_idx = 0
    exn_tab_iter = iter(exn_tab)
    try:

        def step():
            nonlocal end_offset_idx
            entry = next(exn_tab_iter)
            while end_offset_idx < len(offsets) and offsets[end_offset_idx] <= entry.end:
                end_offset_idx += 1
            assert end_offset_idx > 0
            end_offset = offsets[end_offset_idx - 1]
            inst_entry = InstructionExnTabEntry(_get_instruction_by_offset(offset_to_inst, entry.start), _get_instruction_by_offset(offset_to_inst, end_offset), _get_instruction_by_offset(offset_to_inst, entry.target), entry.depth, entry.lasti)
            return (entry, inst_entry)
        entry, inst_entry = step()
        for inst in instructions:
            while inst.offset > entry.end:
                entry, inst_entry = step()
            if inst.offset >= entry.start:
                inst.exn_tab_entry = copy.copy(inst_entry)
    except StopIteration:
        pass