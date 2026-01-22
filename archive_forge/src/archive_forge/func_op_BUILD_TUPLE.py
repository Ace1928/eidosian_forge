import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
def op_BUILD_TUPLE(self, inst: dis.Instruction):
    count = inst.arg
    assert isinstance(count, int)
    items = list(reversed([self.pop() for _ in range(count)]))
    op = Op(opname='build_tuple', bc_inst=inst)
    for i, it in enumerate(items):
        op.add_input(str(i), it)
    self.push(op.add_output('out'))