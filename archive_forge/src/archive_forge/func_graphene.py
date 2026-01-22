from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def graphene(formula='C2', a=2.46, size=(1, 1, 1), vacuum=None):
    """Create a graphene monolayer structure."""
    cell = [[a, 0, 0], [-a / 2, a * 3 ** 0.5 / 2, 0], [0, 0, 0]]
    basis = [[0, 0, 0], [2 / 3, 1 / 3, 0]]
    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)
    if vacuum is not None:
        atoms.center(vacuum, axis=2)
    atoms = atoms.repeat(size)
    return atoms