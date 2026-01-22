from textwrap import dedent
from itertools import islice, product
from sympy.core.basic import Basic
from sympy.core.numbers import Integer
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices.dense import Matrix
from sympy.combinatorics import RGS_enum, RGS_unrank, Permutation
from sympy.utilities.iterables import (
from sympy.utilities.enumerative import (
from sympy.core.singleton import S
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_ordered():
    assert list(ordered((x, y), hash, default=False)) in [[x, y], [y, x]]
    assert list(ordered((x, y), hash, default=False)) == list(ordered((y, x), hash, default=False))
    assert list(ordered((x, y))) == [x, y]
    seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]], (lambda x: len(x), lambda x: sum(x))]
    assert list(ordered(seq, keys, default=False, warn=False)) == [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]
    raises(ValueError, lambda: list(ordered(seq, keys, default=False, warn=True)))