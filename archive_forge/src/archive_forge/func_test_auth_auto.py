import os
import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import pytest
from kivy.network.urlrequest import UrlRequestUrllib as UrlRequest
@pytest.mark.skipif(os.environ.get('NONETWORK'), reason='No network')
def test_auth_auto(kivy_clock):
    obj = UrlRequestQueue([])
    queue = obj.queue
    req = UrlRequest('http://user:passwd@httpbin.org/basic-auth/user/passwd', on_success=obj._on_success, on_progress=obj._on_progress, on_error=obj._on_error, on_redirect=obj._on_redirect, debug=True)
    wait_request_is_finished(kivy_clock, req, timeout=60)
    if req.error and req.error.errno == 11001:
        pytest.skip('Cannot connect to get address')
    ensure_called_from_thread(queue)
    check_queue_values(queue)
    assert queue[-1][2] == ({'authenticated': True, 'user': 'user'},)