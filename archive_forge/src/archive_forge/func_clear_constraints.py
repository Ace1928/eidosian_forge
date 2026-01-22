from ase.gui.i18n import _
import ase.gui.ui as ui
def clear_constraints(self):
    for atoms in self.gui.images:
        atoms.constraints = []
    self.gui.draw()