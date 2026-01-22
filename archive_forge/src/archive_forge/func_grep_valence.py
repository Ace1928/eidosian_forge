import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
def grep_valence(pseudopotential):
    """
    Given a UPF pseudopotential file, find the number of valence atoms.

    Parameters
    ----------
    pseudopotential: str
        Filename of the pseudopotential.

    Returns
    -------
    valence: float
        Valence as reported in the pseudopotential.

    Raises
    ------
    ValueError
        If valence cannot be found in the pseudopotential.
    """
    with open(pseudopotential) as psfile:
        for line in psfile:
            if 'z valence' in line.lower():
                return float(line.split()[0])
            elif 'z_valence' in line.lower():
                if line.split()[0] == '<PP_HEADER':
                    line = list(filter(lambda x: 'z_valence' in x, line.split(' ')))[0]
                return float(line.split('=')[-1].strip().strip('"'))
        else:
            raise ValueError('Valence missing in {}'.format(pseudopotential))