import numpy as np
from pytest import mark
from ase import Atoms
def set_positions_to_orig(atoms, box_len, dimer_separation):
    pos0 = 0.5 * np.full(3, 0.5 * box_len)
    displacement = np.array([0.5 * dimer_separation, 0, 0])
    pos1 = pos0 - displacement
    pos2 = pos0 + displacement
    atoms.set_positions([pos1, pos2])