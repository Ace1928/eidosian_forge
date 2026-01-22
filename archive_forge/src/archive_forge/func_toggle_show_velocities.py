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
def toggle_show_velocities(self, key=None):
    self.draw()