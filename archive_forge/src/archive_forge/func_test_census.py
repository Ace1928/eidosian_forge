from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def test_census(name, census):
    manifolds = [M for M in census]
    print('Checking gluing equations for %d %s manifolds' % (len(manifolds), name))
    max_error = pari(0)
    for i, M in enumerate(manifolds):
        max_error = max(max_error, test_manifold(M))
        print('\r   ' + repr((i, M)).ljust(35) + '   Max error so far: %.2g' % float(max_error), end='')
    print()