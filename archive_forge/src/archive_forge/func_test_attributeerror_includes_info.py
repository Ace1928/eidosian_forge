import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
@pytest.mark.parametrize('name', ['object', 'bool', 'float', 'complex', 'str', 'int'])
def test_attributeerror_includes_info(self, name):
    msg = f'.*\n`np.{name}` was a deprecated alias for the builtin'
    with pytest.raises(AttributeError, match=msg):
        getattr(np, name)