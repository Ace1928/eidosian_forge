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
def test_sequence_partitions():
    assert list(sequence_partitions([1], 1)) == [[[1]]]
    assert list(sequence_partitions([1, 2], 1)) == [[[1, 2]]]
    assert list(sequence_partitions([1, 2], 2)) == [[[1], [2]]]
    assert list(sequence_partitions([1, 2, 3], 1)) == [[[1, 2, 3]]]
    assert list(sequence_partitions([1, 2, 3], 2)) == [[[1], [2, 3]], [[1, 2], [3]]]
    assert list(sequence_partitions([1, 2, 3], 3)) == [[[1], [2], [3]]]
    assert list(sequence_partitions([], 0)) == []
    assert list(sequence_partitions([], 1)) == []
    assert list(sequence_partitions([1, 2], 0)) == []
    assert list(sequence_partitions([1, 2], 3)) == []