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
def get_valence_electrons(symbol, data, pseudo=None):
    """The number of valence electrons for a atomic symbol.

    Parameters
    ----------
    symbol : str
        Chemical symbol

    data : Namelist
        Namelist representing the quantum espresso input parameters

    pseudo : str, optional
        File defining the pseudopotential to be used. If missing a fallback
        to the number of valence electrons recommended at
        http://materialscloud.org/sssp/ is employed.
    """
    if pseudo is None:
        pseudo = '{}_dummy.UPF'.format(symbol)
    for pseudo_dir in get_pseudo_dirs(data):
        if path.exists(path.join(pseudo_dir, pseudo)):
            valence = grep_valence(path.join(pseudo_dir, pseudo))
            break
    else:
        valence = SSSP_VALENCE[atomic_numbers[symbol]]
    return valence