from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
@onlyif_cmds_exist('latex', 'dvipng')
@pytest.mark.parametrize('s, wrap', [('$$x^2$$', False), ('x^2', True)])
def test_latex_to_png_dvipng_runs(s, wrap):
    """
    Test that latex_to_png_dvipng just runs without error.
    """

    def mock_kpsewhich(filename):
        assert filename == 'breqn.sty'
        return None
    latextools.latex_to_png_dvipng(s, wrap)
    with patch_latextool(mock_kpsewhich):
        latextools.latex_to_png_dvipng(s, wrap)