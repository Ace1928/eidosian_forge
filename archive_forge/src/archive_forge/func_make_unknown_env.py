import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
@pytest.fixture
def make_unknown_env():
    varname = 'MODIN_UNKNOWN'
    os.environ[varname] = 'foo'
    yield varname
    del os.environ[varname]