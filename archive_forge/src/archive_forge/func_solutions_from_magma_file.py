from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def solutions_from_magma_file(filename, numerical=False):
    """
    Obsolete, use processFileDispatch.parse_solutions_from_file instead.

    Reads the output from a magma computation from the file with the given
    filename and returns a list of solutions. Also see solutions_from_magma.
    A non-zero dimensional component of the variety is reported as
    NonZeroDimensionalComponent.
    """
    return solutions_from_magma(open(filename).read(), numerical)