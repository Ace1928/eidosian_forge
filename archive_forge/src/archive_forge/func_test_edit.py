import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
def test_edit(self):
    self.ds_version.api.client.patch = self._get_mock_method()
    self._resp.status_code = 202
    self.ds_version.edit('ds-version-1', image='new-image-id')
    self.assertEqual('/mgmt/datastore-versions/ds-version-1', self._url)
    self.assertEqual({'image': 'new-image-id'}, self._body)
    self._resp.status_code = 400
    self.assertRaises(Exception, self.ds_version.edit, 'ds-version-1', 'new-mgr', 'non-existent-image')