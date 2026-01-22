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
def set_frame(self, frame=None, focus=False):
    if frame is None:
        frame = self.frame
    assert frame < len(self.images)
    self.frame = frame
    self.set_atoms(self.images[frame])
    fname = self.images.filenames[frame]
    if fname is None:
        title = 'ase.gui'
    else:
        title = basename(fname)
    self.window.title = title
    self.call_observers()
    if focus:
        self.focus()
    else:
        self.draw()