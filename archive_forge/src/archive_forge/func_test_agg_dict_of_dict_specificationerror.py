from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('agg', [{'A': {'mean': 'mean', 'sum': 'sum'}}, {'A': {'mean': 'mean', 'sum': 'sum'}, 'B': {'mean2': 'mean', 'sum2': 'sum'}}])
def test_agg_dict_of_dict_specificationerror(cases, agg):
    msg = 'nested renamer is not supported'
    with pytest.raises(pd.errors.SpecificationError, match=msg):
        cases.aggregate(agg)