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
def tensor_heads(s, index_types, symmetry=None, comm=0):
    """
    Returns a sequence of TensorHeads from a string `s`
    """
    if isinstance(s, str):
        names = [x.name for x in symbols(s, seq=True)]
    else:
        raise ValueError('expecting a string')
    thlist = [TensorHead(name, index_types, symmetry, comm) for name in names]
    if len(thlist) == 1:
        return thlist[0]
    return thlist