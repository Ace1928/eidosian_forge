import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
@pytest.fixture
def pspot_tmp_path(tmp_path):
    path = os.path.join(tmp_path, 'ppots')
    os.mkdir(path)
    for el in ase.data.chemical_symbols:
        with open(os.path.join(path, '{0}_test.usp'.format(el)), 'w') as fd:
            fd.write('Fake PPOT')
    return path