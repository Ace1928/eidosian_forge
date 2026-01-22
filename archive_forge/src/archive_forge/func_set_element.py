from functools import partial
from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.gui.widgets import Element
from ase.gui.utils import get_magmoms
def set_element(self, element):
    self.gui.atoms.numbers[self.selection()] = element.Z
    self.gui.draw()