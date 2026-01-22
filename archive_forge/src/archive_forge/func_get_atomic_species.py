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
def get_atomic_species(lines, n_species):
    """Parse atomic species from ATOMIC_SPECIES card.

    Parameters
    ----------
    lines : list[str]
        A list of lines containing the ATOMIC_POSITIONS card.
    n_species : int
        Expected number of atom types. Only this many lines will be parsed.

    Returns
    -------
    species : list[(str, float, str)]

    Raises
    ------
    ValueError
        Any problems parsing the data result in ValueError
    """
    species = None
    trimmed_lines = (line.strip() for line in lines if line.strip() and (not line.startswith('#')))
    for line in trimmed_lines:
        if line.startswith('ATOMIC_SPECIES'):
            if species is not None:
                raise ValueError('Multiple ATOMIC_SPECIES specified')
            species = []
            for _dummy in range(n_species):
                label_weight_pseudo = next(trimmed_lines).split()
                species.append((label_weight_pseudo[0], float(label_weight_pseudo[1]), label_weight_pseudo[2]))
    return species