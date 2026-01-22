from ase.gui.i18n import _, ngettext
import ase.gui.ui as ui
import ase.build as build
from ase.data import reference_states
from ase.gui.widgets import Element, pybutton
from ase.build import {func}
def vacuum_checked(self, *args):
    if self.vacuum_check.var.get():
        self.vacuum.active = True
    else:
        self.vacuum.active = False
    self.make()