import numpy as np
import ase.units
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.data import chemical_symbols
def read_acemolecule_input(filename):
    """Reads a ACE-Molecule input file
    Parameters
    ==========
    filename: ACE-Molecule input file name

    Returns
    =======
    ASE atoms object containing geometry only.
    """
    with open(filename, 'r') as fd:
        for line in fd:
            if len(line.split('GeometryFilename')) > 1:
                geometryfile = line.split()[1]
                break
    atoms = read(geometryfile, format='xyz')
    return atoms