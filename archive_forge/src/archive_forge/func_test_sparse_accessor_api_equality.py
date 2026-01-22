import inspect
import numpy as np
import pandas
import pytest
import modin.pandas as pd
@pytest.mark.parametrize('obj', ['DataFrame', 'Series'])
def test_sparse_accessor_api_equality(obj):
    modin_dir = [x for x in dir(getattr(pd, obj).sparse) if x[0] != '_']
    pandas_dir = [x for x in dir(getattr(pandas, obj).sparse) if x[0] != '_']
    missing_from_modin = set(pandas_dir) - set(modin_dir)
    assert not len(missing_from_modin), 'Differences found in API: {}'.format(len(missing_from_modin))
    extra_in_modin = set(modin_dir) - set(pandas_dir)
    assert not len(extra_in_modin), 'Differences found in API: {}'.format(extra_in_modin)