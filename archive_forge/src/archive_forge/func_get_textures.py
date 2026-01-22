from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.io.pov import write_pov, get_bondpairs
from os import unlink
import numpy as np
def get_textures(self):
    return [self.texture_widget.value] * len(self.gui.atoms)