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
@raising_cb_reg
def test_callbackregistry_custom_exception_handler(monkeypatch, cb, excp):
    monkeypatch.setattr(cbook, '_get_running_interactive_framework', lambda: None)
    with pytest.raises(excp):
        cb.process('foo')