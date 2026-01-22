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
def test_color(gui):
    a = Atoms('C10', magmoms=np.linspace(1, -1, 10))
    a.positions[:] = np.linspace(0, 9, 10)[:, None]
    a.calc = SinglePointCalculator(a, forces=a.positions)
    che = np.linspace(100, 110, 10)
    mask = [0] * 10
    mask[5] = 1
    a.set_array('corehole_energies', np.ma.array(che, mask=mask))
    gui.new_atoms(a)
    c = gui.colors_window()
    c.toggle('force')
    c.toggle('magmom')
    activebuttons = [button.active for button in c.radio.buttons]
    assert activebuttons == [1, 0, 1, 0, 0, 1, 1, 1], activebuttons
    c.toggle('corehole_energies')
    c.change_mnmx(101, 120)