import contextlib
import pytest
from panel.tests.conftest import (  # noqa
@pytest.fixture
def bokeh_backend():
    import holoviews as hv
    hv.renderer('bokeh')
    prev_backend = hv.Store.current_backend
    hv.Store.current_backend = 'bokeh'
    yield
    hv.Store.current_backend = prev_backend