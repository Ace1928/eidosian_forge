import inspect
import pytest
from pandas import (
from pandas.core.groupby.generic import (
def test_series_consistency(request, groupby_func):
    if groupby_func in ('first', 'last'):
        msg = 'first and last are entirely different between Series and groupby'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    if groupby_func in ('cumcount', 'corrwith'):
        msg = 'Series has no such method'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    if groupby_func == 'ngroup':
        assert not hasattr(Series, groupby_func)
        return
    series_method = getattr(Series, groupby_func)
    gb_method = getattr(SeriesGroupBy, groupby_func)
    result = set(inspect.signature(gb_method).parameters)
    if groupby_func == 'size':
        expected = {'self'}
    else:
        expected = set(inspect.signature(series_method).parameters)
    exclude_expected, exclude_result = (set(), set())
    if groupby_func in ('any', 'all'):
        exclude_expected = {'kwargs', 'bool_only', 'axis'}
    elif groupby_func in ('diff',):
        exclude_result = {'axis'}
    elif groupby_func in ('max', 'min'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
        exclude_result = {'min_count', 'engine', 'engine_kwargs'}
    elif groupby_func in ('mean', 'std', 'sum', 'var'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
        exclude_result = {'engine', 'engine_kwargs'}
    elif groupby_func in ('median', 'prod', 'sem'):
        exclude_expected = {'axis', 'kwargs', 'skipna'}
    elif groupby_func in ('backfill', 'bfill', 'ffill', 'pad'):
        exclude_expected = {'downcast', 'inplace', 'axis'}
    elif groupby_func in ('cummax', 'cummin'):
        exclude_expected = {'skipna', 'args'}
        exclude_result = {'numeric_only'}
    elif groupby_func in ('cumprod', 'cumsum'):
        exclude_expected = {'skipna'}
    elif groupby_func in ('pct_change',):
        exclude_expected = {'kwargs'}
        exclude_result = {'axis'}
    elif groupby_func in ('rank',):
        exclude_expected = {'numeric_only'}
    elif groupby_func in ('idxmin', 'idxmax'):
        exclude_expected = {'args', 'kwargs'}
    elif groupby_func in ('quantile',):
        exclude_result = {'numeric_only'}
    assert result & exclude_result == exclude_result
    assert expected & exclude_expected == exclude_expected
    result -= exclude_result
    expected -= exclude_expected
    assert result == expected