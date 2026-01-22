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
@pytest.mark.parametrize(('uri', 'expected_resp_status', 'expected_resp_body'), (('/wrong_cl_buffered', 500, b'The requested resource returned more bytes than the declared Content-Length.'), ('/wrong_cl_unbuffered', 200, b'I too')))
def test_Content_Length_out(test_client, uri, expected_resp_status, expected_resp_body):
    """Test response with Content-Length less than the response body.

    (non-chunked response)
    """
    conn = test_client.get_connection()
    conn.putrequest('GET', uri, skip_host=True)
    conn.putheader('Host', conn.host)
    conn.endheaders()
    response = conn.getresponse()
    status_line, _actual_headers, actual_resp_body = webtest.shb(response)
    actual_status = int(status_line[:3])
    assert actual_status == expected_resp_status
    assert actual_resp_body == expected_resp_body
    conn.close()
    test_client.server_instance.error_log.ignored_msgs.extend(("ValueError('Response body exceeds the declared Content-Length.')", "ValueError('Response body exceeds the declared Content-Length.',)"))