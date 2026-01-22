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
@pytest.mark.parametrize('indices, expected_mask', [([1], [False, True]), (None, [True, True])])
def test_dict_indices(self, n2_vibdata, indices, expected_mask):
    vib_data_dict = n2_vibdata.todict()
    vib_data_dict['indices'] = indices
    if indices is not None:
        n_active = len(indices)
        vib_data_dict['hessian'] = np.asarray(vib_data_dict['hessian'])[:n_active, :, :n_active, :].tolist()
    vib_data_fromdict = VibrationsData.fromdict(vib_data_dict)
    assert_array_almost_equal(vib_data_fromdict.get_mask(), expected_mask)