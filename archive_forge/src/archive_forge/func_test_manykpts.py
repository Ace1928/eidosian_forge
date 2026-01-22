import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree
@pytest.mark.skip('expensive')
@calc('abinit')
def test_manykpts(factory):
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.symbols[:2] = 'Cu'
    atoms.calc = factory.calc(nbands=len(atoms) * 7, kpts=[8, 8, 8])
    run(atoms, 'manykpts')