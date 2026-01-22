import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def do_change(remain):
    while remain:
        ent = remain.pop()
        if ent['kind'] == BlockKind('TRY'):
            self.current_block = block
            oldbody = list(block.body)
            block.body.clear()
            self._insert_try_block_end()
            block.body.extend(oldbody)
            return True