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
def test_zero_mass(self, n2_data):
    atoms = n2_data['atoms']
    atoms.set_masses([0.0, 1.0])
    vib_data = VibrationsData(atoms, n2_data['hessian'])
    with pytest.raises(ValueError):
        vib_data.get_energies_and_modes()