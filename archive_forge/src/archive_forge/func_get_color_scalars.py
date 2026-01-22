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
def get_color_scalars(self, frame=None):
    if self.colormode == 'tag':
        return self.atoms.get_tags()
    if self.colormode == 'force':
        f = (self.get_forces() ** 2).sum(1) ** 0.5
        return f * self.images.get_dynamic(self.atoms)
    elif self.colormode == 'velocity':
        return (self.atoms.get_velocities() ** 2).sum(1) ** 0.5
    elif self.colormode == 'initial charge':
        return self.atoms.get_initial_charges()
    elif self.colormode == 'magmom':
        return get_magmoms(self.atoms)
    elif self.colormode == 'neighbors':
        from ase.neighborlist import NeighborList
        n = len(self.atoms)
        nl = NeighborList(self.get_covalent_radii(self.atoms) * 1.5, skin=0, self_interaction=False, bothways=True)
        nl.update(self.atoms)
        return [len(nl.get_neighbors(i)[0]) for i in range(n)]
    else:
        scalars = np.array(self.atoms.get_array(self.colormode), dtype=float)
        return np.ma.array(scalars, mask=np.isnan(scalars))