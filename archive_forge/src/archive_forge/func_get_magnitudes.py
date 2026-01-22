from ase.cell import Cell
from ase.gui.i18n import _
import ase.gui.ui as ui
import numpy as np
def get_magnitudes(self):
    x, y, z = self.cell_grid
    return np.array([x[3].value, y[3].value, z[3].value])