from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
def connected_indices(atoms, index, dmax=None, scale=1.5):
    """Find atoms connected to atoms[index] and return their indices.

    If dmax is not None:
    Atoms are defined to be connected if they are nearer than dmax
    to each other.

    If dmax is None:
    Atoms are defined to be connected if they are nearer than the
    sum of their covalent radii * scale to each other.

    """
    if index < 0:
        index = len(atoms) + index
    if dmax is None:
        radii = scale * covalent_radii[atoms.get_atomic_numbers()]
    else:
        radii = [0.5 * dmax] * len(atoms)
    nl = NeighborList(radii, skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)
    connected = [index] + list(nl.get_neighbors(index)[0])
    isolated = False
    while not isolated:
        isolated = True
        for i in connected:
            for j in nl.get_neighbors(i)[0]:
                if j not in connected:
                    connected.append(j)
                    isolated = False
    return connected