from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def test_polished(dec_prec=200):

    def test_manifold(manifold):
        eqns = manifold.gluing_equations('rect')
        shapes = manifold.tetrahedra_shapes('rect', dec_prec=dec_prec)
        return snap.shapes.gluing_equation_error(eqns, shapes)

    def test_census(name, census):
        manifolds = [M for M in census]
        print('Checking gluing equations for %d %s manifolds' % (len(manifolds), name))
        max_error = pari(0)
        for i, M in enumerate(manifolds):
            max_error = max(max_error, test_manifold(M))
            print('\r   ' + repr((i, M)).ljust(35) + '   Max error so far: %.2g' % float(max_error), end='')
        print()
    test_census('cusped census', snappy.OrientableCuspedCensus(filter='cusps>1')[-100:])
    test_census('closed census', snappy.OrientableClosedCensus()[-100:])
    test_census('4-component links', [M for M in snappy.LinkExteriors(num_cusps=4) if M.solution_type() == 'all tetrahedra positively oriented'])