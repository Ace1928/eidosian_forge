from collections import ChainMap
import inspect
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_mapper_and_positional_arguments_raises(self):
    df = DataFrame([[1]])
    msg = "Cannot specify both 'mapper' and any of 'index' or 'columns'"
    with pytest.raises(TypeError, match=msg):
        df.rename({}, index={})
    with pytest.raises(TypeError, match=msg):
        df.rename({}, columns={})
    with pytest.raises(TypeError, match=msg):
        df.rename({}, columns={}, index={})