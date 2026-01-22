from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_not_float_like(not_floating):
    with pytest.raises(TypeError):
        float_like(not_floating, 'floating')