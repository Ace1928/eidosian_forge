import numpy as np
import pytest
from ase.build import bulk
import ase.units as u
@calc('gpaw', mode={'name': 'pw', 'ecut': 350}, txt=None)
@calc('abinit', chksymtnons=0, ecut=350)
@calc('espresso', tprnfor=True, tstress=True, ecutwfc=350 / u.Ry)
@calc('siesta')
def test_si_stress(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc(kpts=[4, 4, 4])
    atoms.cell[0] *= 0.85
    stress = atoms.get_stress()
    print(stress)
    assert stress == pytest.approx(ref_stress, rel=0.15)
    assert stress[1] == pytest.approx(stress[2], rel=0.01)
    assert stress[4] == pytest.approx(stress[5], rel=0.01)