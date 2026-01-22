from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_not_int_like(not_integer):
    with pytest.raises(TypeError):
        int_like(not_integer, 'integer')