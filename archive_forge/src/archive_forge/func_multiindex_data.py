import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture()
def multiindex_data():
    rng = np.random.default_rng(2)
    ndates = 100
    nitems = 20
    dates = pd.date_range('20130101', periods=ndates, freq='D')
    items = [f'item {i}' for i in range(nitems)]
    data = {}
    for date in dates:
        nitems_for_date = nitems - rng.integers(0, 12)
        levels = [(item, rng.integers(0, 10000) / 100, rng.integers(0, 10000) / 100) for item in items[:nitems_for_date]]
        levels.sort(key=lambda x: x[1])
        data[date] = levels
    return data