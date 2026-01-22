from ase.gui.i18n import _
import numpy as np
import ase.gui.ui as ui
from ase.gui.utils import get_magmoms
def change_mnmx(self, mn=None, mx=None):
    """change min and/or max values for colormap"""
    if mn:
        self.mnmx[1].value = mn
    if mx:
        self.mnmx[3].value = mx
    mn, mx = (self.mnmx[1].value, self.mnmx[3].value)
    colorscale, _, _ = self.gui.colormode_data
    self.gui.colormode_data = (colorscale, mn, mx)
    self.gui.draw()