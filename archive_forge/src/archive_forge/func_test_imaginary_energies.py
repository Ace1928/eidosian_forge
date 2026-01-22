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
def test_imaginary_energies(self, n2_unstable_data):
    vib_data = VibrationsData(n2_unstable_data['atoms'], n2_unstable_data['hessian'])
    assert vib_data.tabulate() == '\n'.join(VibrationsData._tabulate_from_energies(vib_data.get_energies())) + '\n'