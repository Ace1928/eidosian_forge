import numpy as np
import time
from ase.atoms import Atoms
from ase.io import read
from ase.units import Bohr
def read_cube_data(filename):
    """Wrapper function to read not only the atoms information from a cube file
    but also the contained volumetric data.
    """
    dct = read(filename, format='cube', read_data=True, full_output=True)
    return (dct['data'], dct['atoms'])