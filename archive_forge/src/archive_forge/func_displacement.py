import pytest
import numpy as np
from ase.build import bulk
@pytest.fixture
def displacement(atoms):
    rng = np.random.RandomState(12345)
    return 0.1 * (rng.rand(len(atoms), 3) - 0.5)