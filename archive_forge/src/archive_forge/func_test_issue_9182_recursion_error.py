import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def test_issue_9182_recursion_error(self):
    from numba.types import ListType, Tuple, intp

    @numba.njit
    def _sink(x):
        pass

    @numba.njit(cache=False, parallel=True)
    def _ground_node_rule(clauses, nodes):
        for piter in prange(len(nodes)):
            for clause in clauses:
                clause_type = clause[0]
                clause_variables = clause[2]
                if clause_type == 0:
                    clause_var_1 = clause_variables[0]
                elif len(clause_variables) == 2:
                    clause_var_1, clause_var_2 = (clause_variables[0], clause_variables[1])
                elif len(clause_variables) == 4:
                    pass
                if clause_type == 1:
                    _sink(clause_var_1)
                    _sink(clause_var_2)
    _ground_node_rule.compile((ListType(Tuple([intp, intp, ListType(intp)])), ListType(intp)))