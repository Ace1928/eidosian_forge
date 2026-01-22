import os
from skimage._shared._warnings import expected_warnings
import pytest
@pytest.mark.parametrize('strictness', ['0', 'false', 'False', 'FALSE'])
def test_strict_warning_false(setup, strictness):
    os.environ['SKIMAGE_TEST_STRICT_WARNINGS'] = strictness
    with expected_warnings(['some warnings']):
        pass