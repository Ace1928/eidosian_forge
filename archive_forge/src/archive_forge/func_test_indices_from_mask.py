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
@pytest.mark.parametrize('mask,expected_indices', [([True, True, False, True], [0, 1, 3]), ([False, False], []), ([], []), (np.array([True, True]), [0, 1]), (np.array([False, True, True]), [1, 2]), (np.array([], dtype=bool), [])])
def test_indices_from_mask(self, mask, expected_indices):
    assert VibrationsData.indices_from_mask(mask) == expected_indices