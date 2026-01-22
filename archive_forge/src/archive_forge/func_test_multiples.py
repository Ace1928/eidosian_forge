import pytest
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.calculators.h2morse import (H2Morse,
from ase.vibrations.albrecht import Albrecht
def test_multiples(testdir, rrname, atoms):
    """Run multiple vibrational excitations"""
    om = 1
    gam = 0.1
    with Albrecht(atoms, H2MorseExcitedStates, name=rrname, overlap=True, combinations=2, approximation='Albrecht A', txt=None) as ao:
        aoi = ao.intensity(omega=om, gamma=gam)
    assert len(aoi) == 27