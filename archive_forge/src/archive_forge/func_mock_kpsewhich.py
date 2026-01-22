from contextlib import contextmanager
from unittest.mock import patch
import pytest
from IPython.lib import latextools
from IPython.testing.decorators import (
from IPython.utils.process import FindCmdError
def mock_kpsewhich(filename):
    assert filename == 'breqn.sty'
    return None