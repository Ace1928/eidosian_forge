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
def test_energies_and_modes(self, n2_data, n2_vibdata):
    energies, modes = n2_vibdata.get_energies_and_modes()
    assert_array_almost_equal(n2_data['ref_frequencies'], energies / units.invcm, decimal=5)
    assert_array_almost_equal(n2_data['ref_frequencies'], n2_vibdata.get_energies() / units.invcm, decimal=5)
    assert_array_almost_equal(n2_data['ref_frequencies'], n2_vibdata.get_frequencies(), decimal=5)
    assert n2_vibdata.get_zero_point_energy() == pytest.approx(n2_data['ref_zpe'])
    assert n2_vibdata.tabulate() == '\n'.join(VibrationsData._tabulate_from_energies(energies)) + '\n'
    atoms_with_forces = n2_vibdata.show_as_force(-1, show=False)
    try:
        assert_array_almost_equal(atoms_with_forces.get_forces(), n2_data['ref_forces'])
    except AssertionError:
        assert_array_almost_equal(atoms_with_forces.get_forces(), -n2_data['ref_forces'])