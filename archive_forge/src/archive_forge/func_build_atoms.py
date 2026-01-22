import warnings
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase.io.espresso import label_to_symbol
from ase.utils import reader, writer
def build_atoms():
    atoms = Atoms(symbols=symbols, cell=cell, pbc=pbc, positions=positions)
    if not read_arrays:
        return atoms
    info = {'occupancy': occ, 'bfactor': bfactor, 'residuenames': residuenames, 'atomtypes': atomtypes, 'residuenumbers': residuenumbers}
    for name, array in info.items():
        if len(array) == 0:
            pass
        elif len(array) != len(atoms):
            warnings.warn('Length of {} array, {}, different from number of atoms {}'.format(name, len(array), len(atoms)))
        else:
            atoms.set_array(name, np.array(array))
    return atoms