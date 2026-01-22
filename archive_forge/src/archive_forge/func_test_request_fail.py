from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
@mock.patch('blazarclient.base.adapter.LegacyJsonAdapter.request')
def test_request_fail(self, m):
    resp = mock.Mock()
    resp.status_code = 400
    body = {'error message': 'error'}
    m.return_value = (resp, body)
    url = '/leases'
    kwargs = {'body': {'req_key': 'req_value'}}
    self.assertRaises(exception.BlazarClientException, self.manager.request, url, 'POST', **kwargs)