import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_normal_request(test_client):
    """Check that normal GET query succeeds."""
    status_line, _, actual_resp_body = test_client.get('/hello')
    actual_status = int(status_line[:3])
    assert actual_status == HTTP_OK
    assert actual_resp_body == b'Hello world!'