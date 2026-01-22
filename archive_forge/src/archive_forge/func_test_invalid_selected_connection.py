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
def test_invalid_selected_connection(test_client, monkeypatch):
    """Test the error handling segment of HTTP connection selection.

    See :py:meth:`cheroot.connections.ConnectionManager.get_conn`.
    """
    faux_select = FaultySelect(test_client.server_instance._connections._selector.select)
    monkeypatch.setattr(test_client.server_instance._connections._selector, 'select', faux_select)
    faux_get_map = FaultyGetMap(test_client.server_instance._connections._selector._selector.get_map)
    monkeypatch.setattr(test_client.server_instance._connections._selector._selector, 'get_map', faux_get_map)
    resp_status, _resp_headers, _resp_body = test_client.request('/page1', headers=[('Connection', 'Keep-Alive')])
    assert resp_status == '200 OK'
    faux_get_map.sabotage_conn = faux_select.request_served = True
    time.sleep(test_client.server_instance.expiration_interval * 2)
    assert faux_select.os_error_triggered
    assert faux_get_map.conn_closed