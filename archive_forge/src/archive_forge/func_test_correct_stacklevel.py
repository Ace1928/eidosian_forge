import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
def test_correct_stacklevel(self):
    with pytest.warns(UserWarning, match='passes') as record:
        self.raise_warning('passes', UserWarning, stacklevel=2)
    assert_stacklevel(record)