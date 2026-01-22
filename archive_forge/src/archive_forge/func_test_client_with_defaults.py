import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
@pytest.fixture
def test_client_with_defaults(testing_server_with_defaults):
    """Get and return a test client out of the given server."""
    return testing_server_with_defaults.server_client