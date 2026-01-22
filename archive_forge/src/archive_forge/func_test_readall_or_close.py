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
@pytest.mark.parametrize('max_request_body_size', (0, 1001))
def test_readall_or_close(test_client, max_request_body_size):
    """Test a max_request_body_size of 0 (the default) and 1001."""
    old_max = test_client.server_instance.max_request_body_size
    test_client.server_instance.max_request_body_size = max_request_body_size
    conn = test_client.get_connection()
    conn.putrequest('POST', '/err_before_read', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Content-Type', 'text/plain')
    conn.putheader('Content-Length', '1000')
    conn.putheader('Expect', '100-continue')
    conn.endheaders()
    response = conn.response_class(conn.sock, method='POST')
    _version, status, _reason = response._read_status()
    assert status == 100
    skip = True
    while skip:
        skip = response.fp.readline().strip()
    conn.send(b'x' * 1000)
    response.begin()
    status_line, _actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 500
    conn._output(b'POST /upload HTTP/1.1')
    conn._output(('Host: %s' % conn.host).encode('ascii'))
    conn._output(b'Content-Type: text/plain')
    conn._output(b'Content-Length: 17')
    conn._output(b'Expect: 100-continue')
    conn._send_output()
    response = conn.response_class(conn.sock, method='POST')
    version, status, reason = response._read_status()
    assert status == 100
    skip = True
    while skip:
        skip = response.fp.readline().strip()
    body = b'I am a small file'
    conn.send(body)
    response.begin()
    status_line, actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    expected_resp_body = ("thanks for '%s'" % body).encode()
    assert actual_resp_body == expected_resp_body
    conn.close()
    test_client.server_instance.max_request_body_size = old_max