import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_add_security_service_check(self):
    current_security_service = self._FakeSecurityService()
    share_nw = self._FakeShareNetwork()
    expected_path = 'add_security_service_check'
    expected_body = {'security_service_id': current_security_service.id, 'reset_operation': False}
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        self.manager.add_security_service_check(share_nw, current_security_service, False)
        self.manager._action.assert_called_once_with(expected_path, share_nw, expected_body)