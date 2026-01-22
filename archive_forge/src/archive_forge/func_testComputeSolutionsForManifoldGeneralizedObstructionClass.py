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
def testComputeSolutionsForManifoldGeneralizedObstructionClass(manifold, N, compute_solutions, baseline_volumes, baseline_dimensions):
    varieties = manifold.ptolemy_variety(N, obstruction_class='all_generalized')
    assert len(varieties) == 2
    if compute_solutions:

        def compute(variety):
            return variety.compute_solutions()
    else:

        def compute(variety):
            return compute_using_precomputed_magma(variety, dir=testing_files_generalized_directory)
    solutions_trivial = compute(varieties[0])
    solutions_non_trivial = sum([compute(variety) for variety in varieties[1:]], [])
    checkSolutionsForManifoldGeneralizedObstructionClass(solutions_trivial, solutions_non_trivial, manifold, N, baseline_volumes, baseline_dimensions)