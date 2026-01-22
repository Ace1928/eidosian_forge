from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
@pytest.mark.parametrize('idx,nm,prop', [(pd.Index([1]), 'index', 'name'), (pd.Index([1], name='myname'), 'myname', 'name'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')]), ['level_0', 'level_1'], 'names'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')], names=['n1', 'n2']), ['n1', 'n2'], 'names'), (pd.MultiIndex.from_product([('a', 'b'), ('c', 'd')], names=['n1', None]), ['n1', 'level_1'], 'names')])
def test_set_names_unset(self, idx, nm, prop):
    data = pd.Series(1, idx)
    result = set_default_names(data)
    assert getattr(result.index, prop) == nm