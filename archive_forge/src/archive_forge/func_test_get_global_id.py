import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_get_global_id(self):
    global_id = 'req-%s' % uuid.uuid4()
    cl = get_authed_client(global_request_id=global_id)

    @mock.patch.object(requests, 'request', mock_request)
    @mock.patch('time.time', mock.Mock(return_value=1234))
    def test_get_call():
        resp, body = cl.get('/hi')
        headers = {'X-Auth-Token': 'token', 'X-Auth-Project-Id': 'project_id', 'X-OpenStack-Request-ID': global_id, 'User-Agent': cl.USER_AGENT, 'Accept': 'application/json'}
        mock_request.assert_called_with('GET', 'http://example.com/hi', headers=headers, **self.TEST_REQUEST_BASE)
        self.assertEqual({'hi': 'there'}, body)
    test_get_call()