import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_http_connect_request(test_client):
    """Check that CONNECT query results in Method Not Allowed status."""
    status_line = test_client.connect('/anything')[0]
    actual_status = int(status_line[:3])
    assert actual_status == 405