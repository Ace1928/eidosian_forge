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
def build_rvsdg(code, argnames: tuple[str, ...]) -> SCFG:
    byteflow = ByteFlow.from_bytecode(code)
    bcmap = byteflow.scfg.bcmap_from_bytecode(byteflow.bc)
    _scfg_add_conditional_pop_stack(bcmap, byteflow.scfg)
    byteflow = byteflow.restructure()
    canonicalize_scfg(byteflow.scfg)
    if DEBUG_GRAPH:
        render_scfg(byteflow)
    rvsdg = convert_to_dataflow(byteflow, argnames)
    rvsdg = propagate_states(rvsdg)
    if DEBUG_GRAPH:
        from .regionrenderer import RVSDGRenderer, to_graphviz
        to_graphviz(RVSDGRenderer().render(rvsdg)).view('rvsdg')
    return rvsdg