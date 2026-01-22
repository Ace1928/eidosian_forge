import os
import re
import warnings
import numpy as np
from copy import deepcopy
import ase
from ase.parallel import paropen
from ase.spacegroup import Spacegroup
from ase.geometry.cell import cellpar_to_cell
from ase.constraints import FixAtoms, FixedPlane, FixedLine, FixCartesian
from ase.utils import atoms_to_spglib_cell
import ase.units
def read_phonon(filename, index=None, read_vib_data=False, gamma_only=True, frequency_factor=None, units=units_CODATA2002):
    """
    Wrapper function for the more generic read() functionality.

    Note that this is function is intended to maintain backwards-compatibility
    only. For documentation see read_castep_phonon().
    """
    from ase.io import read
    if read_vib_data:
        full_output = True
    else:
        full_output = False
    return read(filename, index=index, format='castep-phonon', full_output=full_output, read_vib_data=read_vib_data, gamma_only=gamma_only, frequency_factor=frequency_factor, units=units)