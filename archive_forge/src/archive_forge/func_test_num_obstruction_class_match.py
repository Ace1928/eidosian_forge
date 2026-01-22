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
def test_num_obstruction_class_match():
    from snappy import OrientableCuspedCensus
    for M in list(OrientableCuspedCensus()[0:5]) + list(OrientableCuspedCensus()[10000:10005]):
        N = NTriangulationForPtolemy(M._to_string())
        assert len(M.ptolemy_obstruction_classes()) == len(N.ptolemy_obstruction_classes())
        for i in range(2, 6):
            assert len(M.ptolemy_generalized_obstruction_classes(i)) == len(N.ptolemy_generalized_obstruction_classes(i))