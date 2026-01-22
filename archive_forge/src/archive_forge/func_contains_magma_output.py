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
def contains_magma_output(text):
    return 'IDEAL=DECOMPOSITION=BEGINS' in text or 'PRIMARY=DECOMPOSITION=BEGINS' in text or 'RADICAL=DECOMPOSITION=BEGINS' in text