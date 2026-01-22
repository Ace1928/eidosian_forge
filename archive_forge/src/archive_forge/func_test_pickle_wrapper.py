from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def test_pickle_wrapper(self):
    fh = BytesIO()
    self.results._results.save(fh)
    fh.seek(0, 0)
    res_unpickled = self.results._results.__class__.load(fh)
    assert type(res_unpickled) is type(self.results._results)
    fh.seek(0, 0)
    self.results.save(fh)
    fh.seek(0, 0)
    res_unpickled = self.results.__class__.load(fh)
    fh.close()
    assert type(res_unpickled) is type(self.results)
    before = sorted(self.results.__dict__.keys())
    after = sorted(res_unpickled.__dict__.keys())
    assert before == after, 'not equal {!r} and {!r}'.format(before, after)
    before = sorted(self.results._results.__dict__.keys())
    after = sorted(res_unpickled._results.__dict__.keys())
    assert before == after, 'not equal {!r} and {!r}'.format(before, after)
    before = sorted(self.results.model.__dict__.keys())
    after = sorted(res_unpickled.model.__dict__.keys())
    assert before == after, 'not equal {!r} and {!r}'.format(before, after)
    before = sorted(self.results._cache.keys())
    after = sorted(res_unpickled._cache.keys())
    assert before == after, 'not equal {!r} and {!r}'.format(before, after)