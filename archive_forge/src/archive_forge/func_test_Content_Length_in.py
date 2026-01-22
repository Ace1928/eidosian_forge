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
def test_Content_Length_in(test_client):
    """Try a non-chunked request where Content-Length exceeds limit.

    (server.max_request_body_size).
    Assert error before body send.
    """
    conn = test_client.get_connection()
    conn.putrequest('POST', '/upload', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.putheader('Content-Type', 'text/plain')
    conn.putheader('Content-Length', '9999')
    conn.endheaders()
    response = conn.getresponse()
    status_line, _actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == 413
    expected_resp_body = b'The entity sent with the request exceeds the maximum allowed bytes.'
    assert actual_resp_body == expected_resp_body
    conn.close()