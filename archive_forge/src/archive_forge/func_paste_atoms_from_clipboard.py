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
def paste_atoms_from_clipboard(self, event=None):
    try:
        atoms = self.clipboard.get_atoms()
    except Exception as err:
        ui.error('Cannot paste atoms', f'Pasting currently works only with the ASE JSON format.\n\nOriginal error:\n\n{err}')
        return
    if self.atoms == Atoms():
        self.atoms.cell = atoms.cell
        self.atoms.pbc = atoms.pbc
    self.paste_atoms_onto_existing(atoms)