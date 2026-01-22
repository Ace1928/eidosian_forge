import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def legalize_names_with_typemap(names, typemap):
    """ We use ir_utils.legalize_names to replace internal IR variable names
        containing illegal characters (e.g. period) with a legal character
        (underscore) so as to create legal variable names.
        The original variable names are in the typemap so we also
        need to add the legalized name to the typemap as well.
    """
    outdict = legalize_names(names)
    for x, y in outdict.items():
        if x != y:
            typemap[y] = typemap[x]
    return outdict