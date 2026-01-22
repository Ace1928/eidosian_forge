import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def diskcache_manager():
    from dash.long_callback import DiskcacheLongCallbackManager
    return DiskcacheLongCallbackManager()