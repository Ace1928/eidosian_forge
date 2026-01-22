import numpy as np
from pytest import mark
from ase import Atoms
def squeeze_dimer(atoms, d):
    """Squeeze the atoms together by the absolute distance ``d`` (Angstroms)
    """
    pos = atoms.get_positions()
    pos[0] += np.asarray([d, 0, 0])
    atoms.set_positions(pos)