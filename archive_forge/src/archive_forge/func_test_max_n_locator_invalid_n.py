import numpy as np
import pytest
from ...util.locator import MaxNLocator
@pytest.mark.parametrize('n', (0, -1, -2))
def test_max_n_locator_invalid_n(n):
    with pytest.raises(ValueError):
        _ = MaxNLocator(n)