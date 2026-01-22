from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.utils import rotate, irotate
def update_angles(self):
    angles = irotate(self.gui.axes)
    for r, a in zip(self.rotate, angles):
        r.value = a