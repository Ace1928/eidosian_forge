import contextlib
import pytest
from panel.tests.conftest import (  # noqa
@pytest.fixture
def mpl_backend():
    import holoviews as hv
    hv.renderer('matplotlib')
    prev_backend = hv.Store.current_backend
    hv.Store.current_backend = 'matplotlib'
    yield
    hv.Store.current_backend = prev_backend