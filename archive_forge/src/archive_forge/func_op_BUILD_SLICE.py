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
def op_BUILD_SLICE(self, inst: dis.Instruction):
    argc = inst.arg
    if argc == 2:
        tos = self.pop()
        tos1 = self.pop()
        start = tos1
        stop = tos
        step = None
    elif argc == 3:
        tos = self.pop()
        tos1 = self.pop()
        tos2 = self.pop()
        start = tos2
        stop = tos1
        step = tos
    else:
        raise Exception('unreachable')
    op = Op(opname='build_slice', bc_inst=inst)
    op.add_input('start', start)
    op.add_input('stop', stop)
    if step is not None:
        op.add_input('step', step)
    self.push(op.add_output('out'))