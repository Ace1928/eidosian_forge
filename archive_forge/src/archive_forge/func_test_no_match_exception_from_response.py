from unittest import mock
from oslotest import base
from aodhclient import exceptions
def test_no_match_exception_from_response(self):
    resp = mock.MagicMock(status_code=520)
    resp.headers = {'Content-Type': 'text/plain', 'x-openstack-request-id': 'fake-request-id'}
    resp.text = 'Of course I still love you'
    e = exceptions.from_response(resp, 'http://no.where:2333/v2/alarms')
    self.assertIsInstance(e, exceptions.ClientException)
    self.assertEqual('Of course I still love you (HTTP 520) (Request-ID: fake-request-id)', '%s' % e)