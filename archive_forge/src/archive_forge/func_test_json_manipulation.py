import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
def test_json_manipulation(self, testdir, random_dimer):
    vib = Vibrations(random_dimer, name='interrupt')
    vib.run()
    disp_file = Path('interrupt/cache.1x-.json')
    comb_file = Path('interrupt/combined.json')
    assert disp_file.is_file()
    assert not comb_file.is_file()
    vib.split()
    assert vib.combine() == 13
    assert not disp_file.is_file()
    assert comb_file.is_file()
    with pytest.raises(RuntimeError):
        vib.run()
    vib.read()
    with open(disp_file, 'w') as fd:
        fd.write('hello')
    with pytest.raises(AssertionError):
        vib.split()
    os.remove(disp_file)
    vib.split()
    assert disp_file.is_file()
    assert not comb_file.is_file()