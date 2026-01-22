import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_add_security_service_to_share_nw_object(self):
    security_service = self._FakeSecurityService()
    share_nw = self._FakeShareNetwork()
    expected_path = 'add_security_service'
    expected_body = {'security_service_id': security_service.id}
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        self.manager.add_security_service(share_nw, security_service)
        self.manager._action.assert_called_once_with(expected_path, share_nw, expected_body)