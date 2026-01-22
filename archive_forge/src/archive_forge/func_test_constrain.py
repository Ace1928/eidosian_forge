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
def test_constrain(gui, atoms):
    gui.select_all()
    dia = gui.constraints_window()
    assert len(atoms.constraints) == 0
    dia.selected()
    assert len(atoms.constraints) == 1
    assert sorted(atoms.constraints[0].index) == list(range(len(atoms)))