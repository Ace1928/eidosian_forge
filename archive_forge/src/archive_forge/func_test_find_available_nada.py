from unittest import mock
from openstack.network.v2 import floating_ip
from openstack import proxy
from openstack.tests.unit import base
def test_find_available_nada(self):
    mock_session = mock.Mock(spec=proxy.Proxy)
    mock_session.default_microversion = None
    fake_response = mock.Mock()
    body = {floating_ip.FloatingIP.resources_key: []}
    fake_response.json = mock.Mock(return_value=body)
    fake_response.status_code = 200
    mock_session.get = mock.Mock(return_value=fake_response)
    self.assertIsNone(floating_ip.FloatingIP.find_available(mock_session))