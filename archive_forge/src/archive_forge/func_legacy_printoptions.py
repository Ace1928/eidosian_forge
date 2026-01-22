import sys
import numpy as np
import pytest
@pytest.fixture(scope='session', autouse=True)
def legacy_printoptions():
    from packaging.version import Version
    if Version(np.__version__) >= Version('1.22'):
        np.set_printoptions(legacy='1.21')