from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@dec.skip_without('pandas')
def test_pandas_series():
    import pandas as pd
    context = limited(data=pd.Series([1], index=['a']))
    assert guarded_eval('data["a"]', context) == 1
    with pytest.raises(KeyError):
        guarded_eval('data["c"]', context)