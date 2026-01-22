import numpy as np
import pytest
import ase.units
from ase.units import CODATA, create_units
import scipy.constants.codata
def test_bad_codata():
    name = 'my_bad_codata_version'
    with pytest.raises(NotImplementedError, match=name):
        create_units(name)