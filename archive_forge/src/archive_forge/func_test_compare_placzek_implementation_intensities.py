import pytest
from pathlib import Path
from ase.parallel import parprint, world
from ase.vibrations.vibrations import Vibrations
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.placzek import Placzek, Profeta
from ase.calculators.h2morse import (H2Morse,
def test_compare_placzek_implementation_intensities(testdir):
    """Intensities of different Placzek implementations
    should be similar"""
    atoms = H2Morse()
    name = 'placzek'
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator, overlap=lambda x, y: x.overlap(y), name=name, txt='-') as rmc:
        rmc.run()
    om = 1
    gam = 0.1
    with Placzek(atoms, H2MorseExcitedStates, name=name, txt=None) as pz:
        pzi = pz.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pzi, 'Placzek')
    with Profeta(atoms, H2MorseExcitedStates, approximation='Placzek', name=name, txt=None) as pr:
        pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pri, 'Profeta using frozenset')
    assert pzi == pytest.approx(pri, 0.001)
    with Profeta(atoms, H2MorseExcitedStates, approximation='Placzek', overlap=True, name=name, txt=None) as pr:
        pro = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pro, 'Profeta using overlap')
    assert pro == pytest.approx(pri, 0.001)