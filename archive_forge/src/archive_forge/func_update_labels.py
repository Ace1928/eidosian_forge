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
def update_labels(self):
    index = self.window['show-labels']
    if index == 0:
        self.labels = None
    elif index == 1:
        self.labels = list(range(len(self.atoms)))
    elif index == 2:
        self.labels = list(get_magmoms(self.atoms))
    elif index == 4:
        Q = self.atoms.get_initial_charges()
        self.labels = ['{0:.4g}'.format(q) for q in Q]
    else:
        self.labels = self.atoms.get_chemical_symbols()