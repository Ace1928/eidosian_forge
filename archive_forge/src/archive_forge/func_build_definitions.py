import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def build_definitions(blocks, definitions=None):
    """Build the definitions table of the given blocks by scanning
    through all blocks and instructions, useful when the definitions
    table is out-of-sync.
    Will return a new definition table if one is not passed.
    """
    if definitions is None:
        definitions = collections.defaultdict(list)
    for block in blocks.values():
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                name = inst.target.name
                definition = definitions.get(name, [])
                if definition == []:
                    definitions[name] = definition
                definition.append(inst.value)
            if type(inst) in build_defs_extensions:
                f = build_defs_extensions[type(inst)]
                f(inst, definitions)
    return definitions