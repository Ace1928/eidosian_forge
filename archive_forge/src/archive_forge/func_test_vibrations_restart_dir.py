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
def test_vibrations_restart_dir(self, testdir, random_dimer):
    vib = Vibrations(random_dimer)
    vib.run()
    freqs = vib.get_frequencies()
    assert freqs is not None
    atoms = random_dimer.copy()
    with ase.utils.workdir('run_from_here', mkdir=True):
        vib = Vibrations(atoms, name=str(Path.cwd().parent / 'vib'))
        assert_array_almost_equal(freqs, vib.get_frequencies())
        assert vib.clean() == 13