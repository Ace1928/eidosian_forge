from math import cos, sin, sqrt
from os.path import basename
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.geometry import complete_cell
from ase.gui.repeat import Repeat
from ase.gui.rotate import Rotate
from ase.gui.render import Render
from ase.gui.colors import ColorWindow
from ase.gui.utils import get_magmoms
from ase.utils import rotate
def get_bonds(atoms, covalent_radii):
    from ase.neighborlist import NeighborList
    nl = NeighborList(covalent_radii * 1.5, skin=0, self_interaction=False)
    nl.update(atoms)
    nbonds = nl.nneighbors + nl.npbcneighbors
    bonds = np.empty((nbonds, 5), int)
    if nbonds == 0:
        return bonds
    n1 = 0
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        n2 = n1 + len(indices)
        bonds[n1:n2, 0] = a
        bonds[n1:n2, 1] = indices
        bonds[n1:n2, 2:] = offsets
        n1 = n2
    i = bonds[:n2, 2:].any(1)
    pbcbonds = bonds[:n2][i]
    bonds[n2:, 0] = pbcbonds[:, 1]
    bonds[n2:, 1] = pbcbonds[:, 0]
    bonds[n2:, 2:] = -pbcbonds[:, 2:]
    return bonds