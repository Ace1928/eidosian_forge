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
def get_precomputed_magma(variety, dir):
    magma_file_name = os.path.join(dir, variety.filename_base() + '.magma_out.bz2')
    return bz2.BZ2File(magma_file_name, 'r').read().decode('ascii')