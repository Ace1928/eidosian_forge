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
def test_progress_iter():
    with capture_output(display=False) as captured:
        for i in display.ProgressBar(5):
            out = captured.stdout
            assert '{0}/5'.format(i) in out
    out = captured.stdout
    assert '5/5' in out