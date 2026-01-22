from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.common.apiclient import exceptions as client_exceptions
from manilaclient import exceptions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import shares
@ddt.data(('2.6', 'os-deny_access'), ('2.7', 'deny_access'))
@ddt.unpack
def test_deny_access_to_share(self, microversion, action_name):
    access_id = 'fake_access_id'
    share = 'fake_share'
    version = api_versions.APIVersion(microversion)
    mock_microversion = mock.Mock(api_version=version)
    manager = shares.ShareManager(api=mock_microversion)
    with mock.patch.object(manager, '_action', mock.Mock(return_value='fake')):
        result = manager.deny(share, access_id)
        manager._action.assert_called_once_with(action_name, share, {'access_id': access_id})
        self.assertEqual('fake', result)