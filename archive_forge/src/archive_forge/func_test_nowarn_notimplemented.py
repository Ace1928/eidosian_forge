from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_nowarn_notimplemented():
    f = HTMLFormatter()

    class HTMLNotImplemented(object):

        def _repr_html_(self):
            raise NotImplementedError
    h = HTMLNotImplemented()
    with capture_output() as captured:
        result = f(h)
    assert result is None
    assert '' == captured.stderr
    assert '' == captured.stdout