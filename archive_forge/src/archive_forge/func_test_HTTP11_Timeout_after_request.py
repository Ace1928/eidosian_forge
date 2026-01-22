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
def test_HTTP11_Timeout_after_request(test_client):
    """Check timeout after at least one request has succeeded.

    The server should close the connection without 408.
    """
    fail_msg = "Writing to timed out socket didn't fail as it should have: %s"
    conn = test_client.get_connection()
    conn.putrequest('GET', '/timeout?t=%s' % timeout, skip_host=True)
    conn.putheader('Host', conn.host)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    assert response.status == 200
    actual_body = response.read()
    expected_body = str(timeout).encode()
    assert actual_body == expected_body
    conn._output(b'GET /hello HTTP/1.1')
    conn._output(('Host: %s' % conn.host).encode('ascii'))
    conn._send_output()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    assert response.status == 200
    actual_body = response.read()
    expected_body = b'Hello, world!'
    assert actual_body == expected_body
    time.sleep(timeout * 2)
    conn._output(b'GET /hello HTTP/1.1')
    conn._output(('Host: %s' % conn.host).encode('ascii'))
    conn._send_output()
    response = conn.response_class(conn.sock, method='GET')
    try:
        response.begin()
    except (socket.error, http.client.BadStatusLine):
        pass
    except Exception as ex:
        pytest.fail(fail_msg % ex)
    else:
        if response.status != 408:
            pytest.fail(fail_msg % response.read())
    conn.close()
    conn = test_client.get_connection()
    conn.putrequest('GET', '/pov', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    assert response.status == 200
    actual_body = response.read()
    expected_body = pov.encode()
    assert actual_body == expected_body
    conn.send(b'GET /hello HTTP/1.1')
    time.sleep(timeout * 2)
    response = conn.response_class(conn.sock, method='GET')
    try:
        response.begin()
    except (socket.error, http.client.BadStatusLine):
        pass
    except Exception as ex:
        pytest.fail(fail_msg % ex)
    else:
        if response.status != 408:
            pytest.fail(fail_msg % response.read())
    conn.close()
    conn = test_client.get_connection()
    conn.putrequest('GET', '/pov', skip_host=True)
    conn.putheader('Host', conn.host)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    assert response.status == 200
    actual_body = response.read()
    expected_body = pov.encode()
    assert actual_body == expected_body
    conn.close()