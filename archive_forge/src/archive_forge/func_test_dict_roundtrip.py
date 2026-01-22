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
def test_dict_roundtrip(self, n2_vibdata):
    vib_data_dict = n2_vibdata.todict()
    vib_data_roundtrip = VibrationsData.fromdict(vib_data_dict)
    for getter in ('get_atoms',):
        assert getattr(n2_vibdata, getter)() == getattr(vib_data_roundtrip, getter)()
    for array_getter in ('get_hessian', 'get_hessian_2d', 'get_mask', 'get_indices'):
        assert_array_almost_equal(getattr(n2_vibdata, array_getter)(), getattr(vib_data_roundtrip, array_getter)())