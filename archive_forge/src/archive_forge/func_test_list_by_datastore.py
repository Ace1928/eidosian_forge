from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_list_by_datastore(self):
    page_mock = mock.Mock()
    self.backups._paginated = page_mock
    limit = 'test-limit'
    marker = 'test-marker'
    datastore = 'test-mysql'
    self.backups.list(limit, marker, datastore)
    page_mock.assert_called_with('/backups', 'backups', limit, marker, {'datastore': datastore})