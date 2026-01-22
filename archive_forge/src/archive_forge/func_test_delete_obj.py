from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_backups
def test_delete_obj(self):
    backup = self._FakeShareBackup
    with mock.patch.object(self.manager, '_delete', mock.Mock()):
        self.manager.delete(backup)
        self.manager._delete.assert_called_once_with(share_backups.RESOURCE_PATH % backup.id)