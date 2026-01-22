import pytest
from .consts import SELENIUM_GRID_DEFAULT
@pytest.fixture
def dash_process_server() -> ProcessRunner:
    """Start a Dash server with subprocess.Popen and waitress-serve."""
    with ProcessRunner() as starter:
        yield starter