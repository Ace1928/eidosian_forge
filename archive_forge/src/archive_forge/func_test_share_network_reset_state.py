import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_networks
def test_share_network_reset_state(self):
    share_nw = self._FakeShareNetwork()
    state = 'active'
    expected_path = 'reset_status'
    expected_body = {'status': state}
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        self.manager.reset_state(share_nw, state)
        self.manager._action.assert_called_once_with(expected_path, share_nw, expected_body)