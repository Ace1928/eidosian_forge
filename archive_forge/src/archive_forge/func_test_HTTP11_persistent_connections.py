import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
def test_HTTP11_persistent_connections(test_client):
    """Test persistent HTTP/1.1 connections."""
    http_connection = test_client.get_connection()
    http_connection.auto_open = False
    http_connection.connect()
    status_line, actual_headers, actual_resp_body = test_client.get('/pov', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert not header_exists('Connection', actual_headers)
    status_line, actual_headers, actual_resp_body = test_client.get('/page1', http_conn=http_connection)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert not header_exists('Connection', actual_headers)
    status_line, actual_headers, actual_resp_body = test_client.get('/page2', http_conn=http_connection, headers=[('Connection', 'close')])
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    assert actual_resp_body == pov.encode()
    assert header_has_value('Connection', 'close', actual_headers)
    with pytest.raises(http.client.NotConnected):
        test_client.get('/pov', http_conn=http_connection)