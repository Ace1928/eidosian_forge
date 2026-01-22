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
def test_new_mass(self, n2_data, n2_vibdata):
    original_masses = n2_vibdata.get_atoms().get_masses()
    new_masses = original_masses * 3
    new_vib_data = n2_vibdata.with_new_masses(new_masses)
    assert_array_almost_equal(new_vib_data.get_atoms().get_masses(), new_masses)
    assert_array_almost_equal(n2_vibdata.get_energies() / np.sqrt(3), new_vib_data.get_energies())