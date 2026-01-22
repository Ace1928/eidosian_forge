import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def open_potcar(filename):
    """ Open POTCAR file with transparent decompression if it's an archive (.Z)
    """
    import gzip
    if filename.endswith('R'):
        return open(filename, 'r')
    elif filename.endswith('.Z'):
        return gzip.open(filename)
    else:
        raise ValueError('Invalid POTCAR filename: "%s"' % filename)