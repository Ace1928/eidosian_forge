import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
def reconstruct_ssa(func_ir):
    """Apply SSA reconstruction algorithm on the given IR.

    Produces minimal SSA using Choi et al algorithm.
    """
    func_ir.blocks = _run_ssa(func_ir.blocks)
    return func_ir