import threading
import time
import pytest
from .._compat import IS_MACOS, IS_WINDOWS  # noqa: WPS436
from ..server import Gateway, HTTPServer
from ..testing import (  # noqa: F401  # pylint: disable=unused-import
from ..testing import get_server_client
@pytest.fixture
def http_request_timeout():
    """Return a common HTTP request timeout for tests with queries."""
    computed_timeout = 0.1
    if IS_MACOS:
        computed_timeout *= 2
    if IS_WINDOWS:
        computed_timeout *= 10
    return computed_timeout