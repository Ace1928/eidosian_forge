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
@pytest.mark.parametrize('kwargs,expected', [(dict(atoms=na2, energies=[1.0], modes=np.array([[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])), [na2_image_1])])
def test_get_jmol_images(self, kwargs, expected):
    from ase.calculators.calculator import compare_atoms
    jmol_images = list(VibrationsData._get_jmol_images(**kwargs))
    assert len(jmol_images) == len(expected)
    for image, reference in zip(jmol_images, expected):
        assert compare_atoms(image, reference) == []
        for key, value in reference.info.items():
            if key == 'frequency_cm-1':
                assert float(image.info[key]) == pytest.approx(value, abs=0.1)
            else:
                assert image.info[key] == value