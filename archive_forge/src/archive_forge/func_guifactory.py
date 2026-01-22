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
@pytest.fixture
def guifactory(display):
    guis = []

    def factory(images):
        gui = GUI(images)
        guis.append(gui)
        return gui
    yield factory
    for gui in guis:
        gui.exit()