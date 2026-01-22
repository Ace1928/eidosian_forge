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
def triangulation_from_magma_file(filename):
    """
    Reads the output from a magma computation from the file with the given
    filename and extracts the manifold for which the file contains solutions.
    """
    return processFileBase.get_manifold_from_file(filename)