import ase.gui.ui as ui
from ase.gui.i18n import _
def set_unit_cell(self):
    self.gui.images.repeat_unit_cell()
    for r in self.repeat:
        r.value = 1
    self.gui.set_frame()