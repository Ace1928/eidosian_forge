from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
@pytest.mark.parametrize('pickle', [True, False])
def test_registration_on_non_empty_registry(self, pickle):
    self.is_empty()
    mini_me = Test_callback_registry()
    self.connect(self.signal, mini_me.dummy, pickle)
    mini_me2 = Test_callback_registry()
    self.connect(self.signal, mini_me2.dummy, pickle)
    mini_me2 = Test_callback_registry()
    self.connect(self.signal, mini_me2.dummy, pickle)
    self.is_not_empty()
    assert self.count() == 2
    mini_me = None
    mini_me2 = None
    self.is_empty()