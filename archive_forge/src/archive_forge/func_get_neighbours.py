import numpy as np
from ase.constraints import Filter, FixAtoms
from ase.geometry.cell import cell_to_cellpar
from ase.neighborlist import neighbor_list
def get_neighbours(atoms, r_cut, self_interaction=False, neighbor_list=neighbor_list):
    """Return a list of pairs of atoms within a given distance of each other.

    Uses ase.neighborlist.neighbour_list to compute neighbors.

    Args:
        atoms: ase.atoms object to calculate neighbours for
        r_cut: cutoff radius (float). Pairs of atoms are considered neighbours
            if they are within a distance r_cut of each other (note that this
            is double the parameter used in the ASE's neighborlist module)
        neighbor_list: function (optional). Optionally replace the built-in
            ASE neighbour list with an alternative with the same call
            signature, e.g. `matscipy.neighbours.neighbour_list`.

    Returns: a tuple (i_list, j_list, d_list, fixed_atoms):
        i_list, j_list: i and j indices of each neighbour pair
        d_list: absolute distance between the corresponding pair
        fixed_atoms: indices of any fixed atoms
    """
    if isinstance(atoms, Filter):
        atoms = atoms.atoms
    i_list, j_list, d_list = neighbor_list('ijd', atoms, r_cut)
    if not self_interaction:
        mask = i_list != j_list
        i_list = i_list[mask]
        j_list = j_list[mask]
        d_list = d_list[mask]
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
    return (i_list, j_list, d_list, fixed_atoms)