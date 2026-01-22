import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_levels_merge_equivalence(self, portfolio, household, expected):
    portfolio = portfolio.copy()
    household = household.copy()
    result = merge(household.reset_index(), portfolio.reset_index(), on=['household_id'], how='inner').set_index(['household_id', 'asset_id'])
    tm.assert_frame_equal(result, expected)