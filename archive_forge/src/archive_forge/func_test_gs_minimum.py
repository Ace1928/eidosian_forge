import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
def test_gs_minimum(testdir):
    """Test ground state minimum distance, energy and
    vibrational frequency"""
    atoms = H2Morse()
    assert atoms.get_distance(0, 1) == pytest.approx(Re[0], 1e-12)
    assert atoms.get_potential_energy() == -De[0]
    vib = Vibrations(atoms)
    vib.run()
    assert vib.get_frequencies().real[-1] == pytest.approx(ome[0], 0.01)