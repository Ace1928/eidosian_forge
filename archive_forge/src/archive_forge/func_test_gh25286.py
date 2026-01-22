import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
def test_gh25286(self):
    info = self.module.charint('T')
    assert info == 2