from __future__ import annotations
import pytest
@pytest.fixture(params=['disk', 'tasks'])
def shuffle_method(request):
    import dask
    with dask.config.set({'dataframe.shuffle.method': request.param}):
        yield request.param