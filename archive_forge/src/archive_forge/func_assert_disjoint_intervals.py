from collections import namedtuple
from pyomo.core.expr.numvalue import value as pyo_value
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable, _DynamicDataBase
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.find_nearest_index import (
def assert_disjoint_intervals(intervals):
    """
    This function takes intervals in the form of tuples and makes sure
    that they are disjoint.

    Arguments
    ---------
    intervals: iterable
        Iterable of tuples, each containing the low and high values of an
        interval.

    """
    intervals = list(sorted(intervals))
    for i, (lo, hi) in enumerate(intervals):
        if not lo <= hi:
            raise RuntimeError('Lower endpoint of interval is higher than upper endpoint')
        if i != 0:
            prev_lo, prev_hi = intervals[i - 1]
            if not prev_hi <= lo:
                raise RuntimeError('Intervals %s and %s are not disjoint' % ((prev_lo, prev_hi), (lo, hi)))