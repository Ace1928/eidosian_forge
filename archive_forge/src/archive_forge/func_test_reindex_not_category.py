from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('index_df,index_res,index_exp', [(CategoricalIndex([], categories=['A']), Index(['A']), Index(['A'])), (CategoricalIndex([], categories=['A']), Index(['B']), Index(['B'])), (CategoricalIndex([], categories=['A']), CategoricalIndex(['A']), CategoricalIndex(['A'])), (CategoricalIndex([], categories=['A']), CategoricalIndex(['B']), CategoricalIndex(['B']))])
def test_reindex_not_category(self, index_df, index_res, index_exp):
    df = DataFrame(index=index_df)
    result = df.reindex(index=index_res)
    expected = DataFrame(index=index_exp)
    tm.assert_frame_equal(result, expected)