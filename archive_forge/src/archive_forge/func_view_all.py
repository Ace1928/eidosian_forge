from ase.gui.i18n import _
import ase.gui.ui as ui
def view_all(self):
    self.gui.images.visible[:] = True
    self.gui.draw()