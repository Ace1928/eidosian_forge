import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
@pytest.fixture
def testing_calculator(testing_keywords, tmp_path, pspot_tmp_path):
    castep_path = os.path.join(tmp_path, 'CASTEP')
    os.mkdir(castep_path)
    return Castep(castep_keywords=testing_keywords, directory=castep_path, castep_pp_path=pspot_tmp_path)