import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
@dec.skip_without('matplotlib')
def test_set_matplotlib_close():
    cfg = _get_inline_config()
    cfg.close_figures = False
    with pytest.deprecated_call():
        display.set_matplotlib_close()
    assert cfg.close_figures
    with pytest.deprecated_call():
        display.set_matplotlib_close(False)
    assert not cfg.close_figures