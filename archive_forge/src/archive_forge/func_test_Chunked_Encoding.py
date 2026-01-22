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
@pytest.mark.xfail(reason=unwrap(trim('\n        Headers from earlier request leak into the request\n        line for a subsequent request, resulting in 400\n        instead of 413. See cherrypy/cheroot#69 for details.\n        ')))
def test_Chunked_Encoding(test_client):
    """Test HTTP uploads with chunked transfer-encoding."""
    conn = test_client.get_connection()
    body = b'8;key=value\r\nxx\r\nxxxx\r\n5\r\nyyyyy\r\n0\r\nContent-Type: application/json\r\n\r\n'
    conn.putrequest('POST', '/upload', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Transfer-Encoding', 'chunked')
    conn.putheader('Trailer', 'Content-Type')
    conn.putheader('Content-Length', '3')
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    status_line, _actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 200
    assert status_line[4:] == 'OK'
    expected_resp_body = ("thanks for '%s'" % b'xx\r\nxxxxyyyyy').encode()
    assert actual_resp_body == expected_resp_body
    body = b'\r\n'.join((b'3e3', b'x' * 995, b'0', b'', b''))
    conn.putrequest('POST', '/upload', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Transfer-Encoding', 'chunked')
    conn.putheader('Content-Type', 'text/plain')
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    status_line, actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 413
    conn.close()