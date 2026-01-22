import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import certifi
import pytest
import responses
from kivy.network.urlrequest import UrlRequestRequests as UrlRequest
from requests.auth import HTTPBasicAuth
from responses import matchers
@responses.activate
def test_auth_header(self, kivy_clock):
    _queue = UrlRequestQueue([])
    head = {'Authorization': 'Basic {}'.format(b64encode(b'exampleuser:examplepassword').decode('utf-8'))}
    responses.get(self.url, body='{}', status=400, content_type='application/json', match=[matchers.header_matcher(head)])
    req = UrlRequest(self.url, req_headers=head, on_finish=_queue._on_finish, debug=True, auth=HTTPBasicAuth('exampleuser', 'examplepassword'))
    self.wait_request_is_finished(kivy_clock, req)
    processed_queue = _queue.queue
    assert len(processed_queue) == 1
    self._ensure_called_from_thread(processed_queue)
    self._check_queue_values(processed_queue[0], 'finish')