from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_bool_like(boolean):
    assert isinstance(bool_like(boolean, 'boolean'), bool)
    assert bool_like(None, 'boolean', optional=True) is None
    if isinstance(boolean, bool):
        assert isinstance(bool_like(boolean, 'boolean', strict=True), bool)
    else:
        with pytest.raises(TypeError):
            bool_like(boolean, 'boolean', strict=True)