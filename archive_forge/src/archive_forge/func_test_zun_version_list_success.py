from unittest import mock
from zunclient.tests.unit.v1 import shell_test_base
@mock.patch('zunclient.v1.versions.VersionManager.list')
def test_zun_version_list_success(self, mock_list):
    self._test_arg_success('version-list')
    self.assertTrue(mock_list.called)