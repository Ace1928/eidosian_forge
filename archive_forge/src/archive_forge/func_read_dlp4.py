import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
def read_dlp4(f, symbols=None):
    """Read a DL_POLY_4 config/revcon file.

    Typically used indirectly through read('filename', atoms, format='dlp4').

    Can be unforgiven with custom chemical element names.
    Please complain to alin@elena.space for bugs."""
    line = f.readline()
    line = f.readline().split()
    levcfg = int(line[0])
    imcon = int(line[1])
    position = f.tell()
    line = f.readline()
    tokens = line.split()
    is_trajectory = tokens[0] == 'timestep'
    if not is_trajectory:
        f.seek(position)
    while line:
        if is_trajectory:
            tokens = line.split()
            natoms = int(tokens[2])
        else:
            natoms = None
        yield read_single_image(f, levcfg, imcon, natoms, is_trajectory, symbols)
        line = f.readline()