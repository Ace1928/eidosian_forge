import numpy as np
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
def next_bond(atoms):
    """
    Generates bonds (lazily) one at a time, sorted by k-value (low to high).
    Here, k = d_ij / (r_i + r_j), where d_ij is the bond length and r_i and r_j
    are the covalent radii of atoms i and j.

    Parameters:

    atoms: ASE atoms object

    Returns: iterator of bonds
        A bond is a tuple with the following elements:

        k:       float   k-value
        i:       float   index of first atom
        j:       float   index of second atom
        offset:  tuple   cell offset of second atom
    """
    kmax = 0
    rs = covalent_radii[atoms.get_atomic_numbers()]
    seen = set()
    while 1:
        kmax += 2
        nl = NeighborList(kmax * rs, skin=0, self_interaction=False)
        nl.update(atoms)
        bonds = get_bond_list(atoms, nl, rs)
        new_bonds = bonds - seen
        if len(new_bonds) == 0:
            break
        seen.update(new_bonds)
        for b in sorted(new_bonds, key=lambda x: x[0]):
            yield b