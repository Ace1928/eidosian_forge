from itertools import zip_longest
from sympy.utilities.enumerative import (
from sympy.utilities.iterables import _set_partitions
def subrange_exercise(mult, lb, ub):
    """Compare filter-based and more optimized subrange implementations

    Helper for tests, called with both small and larger multisets.
    """
    m = MultisetPartitionTraverser()
    assert m.count_partitions(mult) == m.count_partitions_slow(mult)
    ma = MultisetPartitionTraverser()
    mc = MultisetPartitionTraverser()
    md = MultisetPartitionTraverser()
    a_it = ma.enum_range(mult, lb, ub)
    b_it = part_range_filter(multiset_partitions_taocp(mult), lb, ub)
    c_it = part_range_filter(mc.enum_small(mult, ub), lb, sum(mult))
    d_it = part_range_filter(md.enum_large(mult, lb), 0, ub)
    for sa, sb, sc, sd in zip_longest(a_it, b_it, c_it, d_it):
        assert compare_multiset_states(sa, sb)
        assert compare_multiset_states(sa, sc)
        assert compare_multiset_states(sa, sd)