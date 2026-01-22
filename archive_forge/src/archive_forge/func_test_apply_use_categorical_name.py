from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_apply_use_categorical_name(df):
    cats = qcut(df.C, 4)

    def get_stats(group):
        return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
    result = df.groupby(cats, observed=False).D.apply(get_stats)
    assert result.index.names[0] == 'C'