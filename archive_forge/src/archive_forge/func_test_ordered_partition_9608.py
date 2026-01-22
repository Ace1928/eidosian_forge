from sympy.core.sorting import ordered, default_sort_key
from sympy.combinatorics.partitions import (Partition, IntegerPartition,
from sympy.testing.pytest import raises
from sympy.utilities.iterables import partitions
from sympy.sets.sets import Set, FiniteSet
def test_ordered_partition_9608():
    a = Partition([1, 2, 3], [4])
    b = Partition([1, 2], [3, 4])
    assert list(ordered([a, b], Set._infimum_key))