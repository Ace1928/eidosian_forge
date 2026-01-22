import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_share_network_security_service_update_check(self):
    current_security_service = self._FakeSecurityService()
    new_security_service = self._FakeSecurityService()
    share_nw = self._FakeShareNetwork()
    expected_path = 'update_security_service_check'
    expected_body = {'current_service_id': current_security_service.id, 'new_service_id': new_security_service.id, 'reset_operation': False}
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        self.manager.update_share_network_security_service_check(share_nw, current_security_service, new_security_service)
        self.manager._action.assert_called_once_with(expected_path, share_nw, expected_body)