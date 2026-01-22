from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def test_optional_dict_like_error():
    match = 'value must be a dict or dict_like \\(i.e., a Mapping\\)'
    with pytest.raises(TypeError, match=match):
        dict_like([], 'value', optional=True)
    with pytest.raises(TypeError, match=match):
        dict_like({'a'}, 'value', optional=True)
    with pytest.raises(TypeError, match=match):
        dict_like('a', 'value', optional=True)