from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
def test_unset_keys_error(self):
    mock_manager_delete = self.mock_object(self.manager, '_delete', mock.Mock(return_value='error'))
    result = self.share_group_type.unset_keys(sorted(self.fake_group_specs.keys()))
    self.assertEqual('error', result)
    mock_manager_delete.assert_called_once_with(types.GROUP_SPECS_RESOURCE_PATH % (fake.ShareGroupType.id, 'key1'))