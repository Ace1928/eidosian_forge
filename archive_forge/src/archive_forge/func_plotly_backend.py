import contextlib
import pytest
from panel.tests.conftest import (  # noqa
@pytest.fixture
def plotly_backend():
    import holoviews as hv
    hv.renderer('plotly')
    prev_backend = hv.Store.current_backend
    hv.Store.current_backend = 'plotly'
    yield
    hv.Store.current_backend = prev_backend