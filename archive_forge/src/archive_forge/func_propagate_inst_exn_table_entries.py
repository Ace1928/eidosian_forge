import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def propagate_inst_exn_table_entries(instructions: List[Instruction]) -> None:
    """
    Copies exception table entries to all instructions in an entry's range.
    Supports nested exception table entries.
    """
    indexof = get_indexof(instructions)
    entries: Dict[Tuple[int, int], InstructionExnTabEntry] = {}
    for inst in instructions:
        if inst.exn_tab_entry:
            key = (indexof[inst.exn_tab_entry.start], indexof[inst.exn_tab_entry.end])
            if key in entries:
                assert inst.exn_tab_entry == entries[key]
            entries[key] = inst.exn_tab_entry
    sorted_entries = [entries[key] for key in sorted(entries.keys(), key=lambda t: (t[0], -t[1]))]
    check_inst_exn_tab_entries_nested(sorted_entries, indexof)
    for entry in sorted_entries:
        for i in range(indexof[entry.start], indexof[entry.end] + 1):
            instructions[i].exn_tab_entry = copy.copy(entry)