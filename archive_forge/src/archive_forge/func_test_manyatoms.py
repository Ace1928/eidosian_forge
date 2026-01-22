import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree
@pytest.mark.skip('expensive')
@calc('abinit')
def test_manyatoms(factory):
    atoms = bulk('Ne', cubic=True) * (4, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.calc = factory.calc(nbands=len(atoms) * 5)
    run(atoms, 'manyatoms')