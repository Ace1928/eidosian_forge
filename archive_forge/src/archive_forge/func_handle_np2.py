import pytest
@pytest.fixture(autouse=True)
def handle_np2():
    try:
        import numpy as np
        np.set_printoptions(legacy='1.21')
    except ImportError:
        pass