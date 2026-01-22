from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_bad_precision():
    """test various invalid values for float_precision."""
    f = PlainTextFormatter()

    def set_fp(p):
        f.float_precision = p
    pytest.raises(ValueError, set_fp, '%')
    pytest.raises(ValueError, set_fp, '%.3f%i')
    pytest.raises(ValueError, set_fp, 'foo')
    pytest.raises(ValueError, set_fp, -1)