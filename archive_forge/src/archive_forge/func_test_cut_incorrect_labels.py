import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('labels', ['foo', 1, True])
def test_cut_incorrect_labels(labels):
    values = range(5)
    msg = 'Bin labels must either be False, None or passed in as a list-like argument'
    with pytest.raises(ValueError, match=msg):
        cut(values, 4, labels=labels)