import inspect
import pydoc
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_tab_completion_with_categorical(self):
    ok_for_cat = ['categories', 'codes', 'ordered', 'set_categories', 'add_categories', 'remove_categories', 'rename_categories', 'reorder_categories', 'remove_unused_categories', 'as_ordered', 'as_unordered']
    s = Series(list('aabbcde')).astype('category')
    results = sorted({r for r in s.cat.__dir__() if not r.startswith('_')})
    tm.assert_almost_equal(results, sorted(set(ok_for_cat)))