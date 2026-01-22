import errno
import socket
import urllib.parse  # noqa: WPS301
import pytest
from cheroot.test import helper
def test_send_header_before_closing(testing_server_close):
    """Test we are actually sending the headers before calling 'close'."""
    _, _, resp_body = testing_server_close.server_client.get('/')
    assert resp_body == b'hello'