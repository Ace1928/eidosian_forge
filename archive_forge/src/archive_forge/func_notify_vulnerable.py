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
def notify_vulnerable(self):
    """Notify windows that would break when new_atoms is called.

        The notified windows may adapt to the new atoms.  If that is not
        possible, they should delete themselves.
        """
    new_vul = []
    for wref in self.vulnerable_windows:
        ref = wref()
        if ref is not None:
            new_vul.append(wref)
            ref.notify_atoms_changed()
    self.vulnerable_windows = new_vul