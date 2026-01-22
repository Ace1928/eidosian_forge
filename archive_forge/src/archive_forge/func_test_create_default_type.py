from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_groups
def test_create_default_type(self):
    fake_share_group = fake.ShareGroup()
    mock_create = self.mock_object(self.manager, '_create', mock.Mock(return_value=fake_share_group))
    create_args = {'name': fake.ShareGroup.name, 'description': fake.ShareGroup.description, 'availability_zone': fake.ShareGroup.availability_zone}
    result = self.manager.create(**create_args)
    self.assertIs(fake_share_group, result)
    expected_body = {share_groups.RESOURCE_NAME: create_args}
    mock_create.assert_called_once_with(share_groups.RESOURCES_PATH, expected_body, share_groups.RESOURCE_NAME)