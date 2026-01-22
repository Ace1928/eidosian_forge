import inspect
import re
import warnings
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import (
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from warnings import warn
@pytest.mark.parametrize('level', [1, 3])
def test_wrong_stacklevel(self, level):
    with pytest.warns(UserWarning, match='wrong') as record:
        self.raise_warning('wrong', UserWarning, stacklevel=level)
    line_number = inspect.currentframe().f_lineno - 2
    regex = '.*' + re.escape(f'!= {__file__}:{line_number}')
    with pytest.raises(AssertionError, match=regex):
        assert_stacklevel(record, offset=-5)