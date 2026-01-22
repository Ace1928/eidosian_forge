import numpy as np
import pytest
from ase import Atom, Atoms
from ase.io.nwchem import write_nwchem_in
@pytest.fixture
def atomic_configuration():
    molecule = Atoms(pbc=False)
    molecule.append(Atom('C', [0, 0, 0]))
    molecule.append(Atom('O', [1.6, 0, 0]))
    return molecule