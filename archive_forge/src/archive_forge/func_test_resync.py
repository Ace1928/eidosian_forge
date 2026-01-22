from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common import constants
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_replicas
def test_resync(self):
    with mock.patch.object(self.manager, '_action', mock.Mock()):
        self.manager.resync(FAKE_REPLICA)
        self.manager._action.assert_called_once_with('resync', FAKE_REPLICA)