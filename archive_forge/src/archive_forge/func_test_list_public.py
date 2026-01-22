from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_type_access as type_access
def test_list_public(self):
    fake_share_group_type_access = fake.ShareGroupTypeAccess()
    mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type_access]))
    fake_share_group_type = fake.ShareGroupType()
    fake_share_group_type.is_public = True
    result = self.manager.list(fake_share_group_type)
    self.assertIsNone(result)
    self.assertFalse(mock_list.called)