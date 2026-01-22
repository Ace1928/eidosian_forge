from snappy import verify, Manifold
from snappy.verify import upper_halfspace, cusp_shapes, cusp_areas, volume
from snappy.sage_helper import _within_sage, doctest_modules
import sys
import getopt
def check_certified_intervals():
    for n in ['m009', 'm015', 't02333', 't02333(1,2)', 'm129(2,3)', 'm129(2,3)(3,4)']:
        M = Manifold(n)
        high_prec = M.tetrahedra_shapes('rect', bits_prec=1000)
        intervals = M.tetrahedra_shapes('rect', bits_prec=100, intervals=True)
        for z, interval in zip(high_prec, intervals):
            if not abs(interval.center() - z) < 1e-10:
                raise Exception
            if z not in interval:
                raise Exception