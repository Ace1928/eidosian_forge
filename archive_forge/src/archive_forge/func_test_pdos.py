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
def test_pdos(self, n2_vibdata):
    with pytest.warns(np.ComplexWarning):
        pdos = n2_vibdata.get_pdos()
    assert_array_almost_equal(pdos[0].get_energies(), n2_vibdata.get_energies())
    assert_array_almost_equal(pdos[1].get_energies(), n2_vibdata.get_energies())
    assert sum(pdos[0].get_weights()) == pytest.approx(3.0)