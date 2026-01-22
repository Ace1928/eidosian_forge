import os
from skimage._shared._warnings import expected_warnings
import pytest
def test_strict_warnigns_default(setup):
    with pytest.raises(ValueError):
        with expected_warnings(['some warnings']):
            pass