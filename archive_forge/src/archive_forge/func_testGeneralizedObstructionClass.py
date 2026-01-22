from snappy import Manifold, pari, ptolemy
from snappy.ptolemy import solutions_from_magma, Flattenings, parse_solutions
from snappy.ptolemy.processFileBase import get_manifold
from snappy.ptolemy import __path__ as ptolemy_paths
from snappy.ptolemy.coordinates import PtolemyCannotBeCheckedError
from snappy.sage_helper import _within_sage, doctest_modules
from snappy.pari import pari
import bz2
import os
import sys
def testGeneralizedObstructionClass(compute_solutions):
    vols = [pari('0'), 2 * vol_tet]
    test__m003__2 = (ManifoldGetter('m003'), 2, vols, [0])
    vols = [2 * vol_tet]
    test__m004__2 = (ManifoldGetter('m004'), 2, vols, [0])
    vols = [pari('0'), pari('2.595387593686742138301993834077989475956329764530314161212797242812715071384508096863829303251915501'), 2 * 4 * vol_tet]
    test__m003__3 = (ManifoldGetter('m003'), 3, vols, [0, 1])
    test_cases = [test__m003__2, test__m004__2]
    if not _within_sage or not compute_solutions:
        test_cases += [test__m003__3]
    for manifold, N, vols, dims in test_cases:
        print('Checking for', manifold.name(), 'N = %d' % N)
        testComputeSolutionsForManifoldGeneralizedObstructionClass(manifold, N, compute_solutions, vols, dims)