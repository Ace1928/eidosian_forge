import os
from pathlib import Path
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import molecule, bulk
import ase.gui.ui as ui
from ase.gui.i18n import _
from ase.gui.gui import GUI
from ase.gui.save import save_dialog
from ase.gui.quickinfo import info
def test_clipboard_cut_paste(gui):
    atoms = molecule('H2O')
    gui.new_atoms(atoms.copy())
    assert len(gui.atoms) == 3
    gui.select_all()
    gui.cut_atoms_to_clipboard()
    assert len(gui.atoms) == 0
    assert atoms == gui.clipboard.get_atoms()