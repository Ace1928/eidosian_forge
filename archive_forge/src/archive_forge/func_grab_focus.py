from ase.gui.i18n import _
import ase.data
import ase.gui.ui as ui
from ase import Atoms
def grab_focus(self):
    self.z_entry.entry.focus_set()