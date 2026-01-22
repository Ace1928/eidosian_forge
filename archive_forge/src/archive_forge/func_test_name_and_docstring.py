import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
def test_name_and_docstring(self):
    assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
    if sys.flags.optimize < 2:
        assert_equal(dispatched_one_arg.__doc__, 'Docstring.')