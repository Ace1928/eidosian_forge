import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dash_thread_server() -> ThreadedRunner:
    """Start a local dash server in a new thread."""
    with ThreadedRunner() as starter:
        yield starter