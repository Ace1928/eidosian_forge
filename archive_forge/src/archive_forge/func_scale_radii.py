from ase.gui.i18n import _
import ase.gui.ui as ui
def scale_radii(self):
    self.gui.images.atom_scale = self.scale.value
    self.gui.draw()
    return True