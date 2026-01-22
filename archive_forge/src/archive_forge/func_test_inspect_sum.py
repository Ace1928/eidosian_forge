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
def test_inspect_sum(self):
    signature = inspect.signature(np.sum)
    assert_('axis' in signature.parameters)