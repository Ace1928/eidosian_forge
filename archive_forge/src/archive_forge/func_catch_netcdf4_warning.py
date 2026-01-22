import numpy as np
import pytest
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory
@pytest.fixture(autouse=True)
def catch_netcdf4_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        yield