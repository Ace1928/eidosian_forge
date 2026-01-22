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
def indices_canon_args(self):
    """
        Returns ``(g, dummies, msym, v)``, the entries of ``canonicalize``

        See ``canonicalize`` in ``tensor_can.py`` in combinatorics module.
        """
    from sympy.combinatorics.permutations import _af_new
    n = self._ext_rank
    g = [None] * n + [n, n + 1]

    def metric_symmetry_to_msym(metric):
        if metric is None:
            return None
        sym = metric.symmetry
        if sym == TensorSymmetry.fully_symmetric(2):
            return 0
        if sym == TensorSymmetry.fully_symmetric(-2):
            return 1
        return None
    for i, (indx, ipos) in enumerate(self._get_sorted_free_indices_for_canon()):
        g[ipos] = i
    pos = len(self.free)
    j = len(self.free)
    dummies = []
    prev = None
    a = []
    msym = []
    for ipos1, ipos2 in self._get_sorted_dum_indices_for_canon():
        g[ipos1] = j
        g[ipos2] = j + 1
        j += 2
        typ = self.index_types[ipos1]
        if typ != prev:
            if a:
                dummies.append(a)
            a = [pos, pos + 1]
            prev = typ
            msym.append(metric_symmetry_to_msym(typ.metric))
        else:
            a.extend([pos, pos + 1])
        pos += 2
    if a:
        dummies.append(a)
    return (_af_new(g), dummies, msym)