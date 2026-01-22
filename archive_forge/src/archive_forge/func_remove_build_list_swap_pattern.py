from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def remove_build_list_swap_pattern(self, entries):
    """ Find the following bytecode pattern:

            BUILD_{LIST, MAP, SET}
            SWAP(2)
            FOR_ITER
            ...
            END_FOR
            SWAP(2)

            This pattern indicates that a list/dict/set comprehension has
            been inlined. In this case we can skip the exception blocks
            entirely along with the dead exceptions that it points to.
            A pair of exception that sandwiches these exception will
            also be merged into a single exception.
        """

    def pop_and_merge_exceptions(entries: list, entry_to_remove: _ExceptionTableEntry):
        lower_entry_idx = entries.index(entry_to_remove) - 1
        upper_entry_idx = entries.index(entry_to_remove) + 1
        if lower_entry_idx >= 0 and upper_entry_idx < len(entries):
            lower_entry = entries[lower_entry_idx]
            upper_entry = entries[upper_entry_idx]
            if lower_entry.target == upper_entry.target:
                entries[lower_entry_idx] = _ExceptionTableEntry(lower_entry.start, upper_entry.end, lower_entry.target, lower_entry.depth, upper_entry.lasti)
                entries.remove(upper_entry)
        entries.remove(entry_to_remove)
        entries = [e for e in entries if not e.start == entry_to_remove.target]
        return entries
    work_remaining = True
    while work_remaining:
        work_remaining = False
        for entry in entries.copy():
            index = self.ordered_offsets.index(entry.start)
            curr_inst = self.table[self.ordered_offsets[index]]
            if curr_inst.opname not in ('BUILD_LIST', 'BUILD_MAP', 'BUILD_SET'):
                continue
            next_inst = self.table[self.ordered_offsets[index + 1]]
            if not next_inst.opname == 'SWAP' and next_inst.arg == 2:
                continue
            next_inst = self.table[self.ordered_offsets[index + 2]]
            if not next_inst.opname == 'FOR_ITER':
                continue
            index = self.ordered_offsets.index(entry.end)
            curr_inst = self.table[self.ordered_offsets[index - 1]]
            if not curr_inst.opname == 'END_FOR':
                continue
            next_inst = self.table[self.ordered_offsets[index]]
            if not next_inst.opname == 'SWAP' and next_inst.arg == 2:
                continue
            entries = pop_and_merge_exceptions(entries, entry)
            work_remaining = True
    return entries