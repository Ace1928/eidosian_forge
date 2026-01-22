import ase.gui.ui as ui
from ase.gui.i18n import _
def unconstrain(self):
    self.gui.images.set_dynamic(self.gui.images.selected, True)
    self.gui.draw()