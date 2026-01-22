import pickle
import subprocess
import sys
import weakref
from functools import partial
from ase.gui.i18n import _
from time import time
import numpy as np
from ase import Atoms, __version__
import ase.gui.ui as ui
from ase.gui.defaults import read_defaults
from ase.gui.images import Images
from ase.gui.nanoparticle import SetupNanoparticle
from ase.gui.nanotube import SetupNanotube
from ase.gui.save import save_dialog
from ase.gui.settings import Settings
from ase.gui.status import Status
from ase.gui.surfaceslab import SetupSurfaceSlab
from ase.gui.view import View
def toggle_arrowkey_mode(self, mode):
    assert mode != self.ARROWKEY_SCAN
    if self.arrowkey_mode == mode:
        self.arrowkey_mode = self.ARROWKEY_SCAN
        self.move_atoms_mask = None
    else:
        self.arrowkey_mode = mode
        self.move_atoms_mask = self.images.selected.copy()
    self.draw()