import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
@contextmanager
def set_block(self, label: int, block: ir.Block) -> Iterator[ir.Block]:
    """A context manager that set the current block for other IR building
        methods.

        In addition,

        - It closes any existing block in ``last_block_label`` by jumping to the
          new block.
        - If there is a existing block, it will be restored as the current block
          after the context manager.
        """
    if self.last_block_label is not None:
        last_block = self.blocks[self.last_block_label]
        if not last_block.is_terminated:
            last_block.append(ir.Jump(label, loc=self.loc))
        if self._emit_debug_print:
            print('begin dump last blk'.center(80, '-'))
            last_block.dump()
            print('end dump last blk'.center(80, '='))
    self.blocks[label] = block
    old = self._current_block
    self._current_block = block
    try:
        yield block
    finally:
        self.last_block_label = label
        self._current_block = old
        if self._emit_debug_print:
            print(f'begin dump blk: {label}'.center(80, '-'))
            block.dump()
            print('end dump blk'.center(80, '='))