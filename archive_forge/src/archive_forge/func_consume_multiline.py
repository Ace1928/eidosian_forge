import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def consume_multiline(fd, headerline, nvalues, dtype):
    """Parse abinit-formatted "header + values" sections.

    Example:

        typat 1 1 1 1 1
              1 1 1 1
    """
    tokens = headerline.split()
    assert tokens[0].isalpha()
    values = tokens[1:]
    while len(values) < nvalues:
        line = next(fd)
        values.extend(line.split())
    assert len(values) == nvalues
    return np.array(values).astype(dtype)