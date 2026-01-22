import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_assignment1(self):
    a = self.data
    assert_equal(a.col1[0], 1)
    a[0].col1 = 0
    assert_equal(a.col1[0], 0)