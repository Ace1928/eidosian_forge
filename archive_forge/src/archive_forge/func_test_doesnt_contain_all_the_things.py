import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_doesnt_contain_all_the_things(self):
    idx = Index([np.nan])
    assert not idx.isin([0]).item()
    assert not idx.isin([1]).item()
    assert idx.isin([np.nan]).item()