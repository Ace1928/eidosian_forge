from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_format_config():
    """config objects don't pretend to support fancy reprs with lazy attrs"""
    f = HTMLFormatter()
    cfg = Config()
    with capture_output() as captured:
        result = f(cfg)
    assert result is None
    assert captured.stderr == ''
    with capture_output() as captured:
        result = f(Config)
    assert result is None
    assert captured.stderr == ''