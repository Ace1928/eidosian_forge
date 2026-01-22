from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
@staticmethod
def generate_indices_from_free_dum_index_types(free, dum, index_types):
    indices = [None] * (len(free) + 2 * len(dum))
    for idx, pos in free:
        indices[pos] = idx
    generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
    for pos1, pos2 in dum:
        typ1 = index_types[pos1]
        indname = generate_dummy_name(typ1)
        indices[pos1] = TensorIndex(indname, typ1, True)
        indices[pos2] = TensorIndex(indname, typ1, False)
    return _IndexStructure._replace_dummy_names(indices, free, dum)