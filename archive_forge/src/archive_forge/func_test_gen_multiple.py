import pytest
import numpy as np
from ase import Atoms
from ase.io import read, write
def test_gen_multiple():
    atoms = Atoms('H2')
    with pytest.raises(ValueError):
        write('test.gen', [atoms, atoms])