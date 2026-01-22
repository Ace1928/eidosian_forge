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
def paste_atoms_onto_existing(self, atoms):
    selection = self.selected_atoms()
    if len(selection):
        paste_center = selection.positions.sum(axis=0) / len(selection)
        atoms = atoms.copy()
        atoms.cell = (1, 1, 1)
        atoms.center(about=paste_center)
    self.add_atoms_and_select(atoms)
    self.move_atoms_mask = self.images.selected.copy()
    self.arrowkey_mode = self.ARROWKEY_MOVE
    self.draw()