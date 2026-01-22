from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_snapshots as snapshots
def test_delete_force(self):
    mock_delete = self.mock_object(self.manager, '_delete')
    mock_post = self.mock_object(self.manager.api.client, 'post')
    self.manager.delete(fake.ShareGroupSnapshot.id, force=True)
    self.assertFalse(mock_delete.called)
    mock_post.assert_called_once_with(snapshots.RESOURCE_PATH_ACTION % fake.ShareGroupSnapshot.id, body={'force_delete': None})