from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
def connected_atoms(atoms, index, dmax=None, scale=1.5):
    """Find all atoms connected to atoms[index] and return them."""
    return atoms[connected_indices(atoms, index, dmax, scale)]