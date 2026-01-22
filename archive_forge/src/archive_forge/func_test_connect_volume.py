import os
import sys
from unittest import mock
import ddt
from glance_store import exceptions
from glance_store.tests.unit.cinder import test_base as test_base_connector
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
@ddt.data((False, 'raw'), (False, 'qcow2'), (True, 'raw'), (True, 'qcow2'))
@ddt.unpack
def test_connect_volume(self, encrypted, file_format):
    fake_vol = mock.MagicMock(id='fake_vol_uuid', encrypted=encrypted)
    fake_attachment = mock.MagicMock(id='fake_attachment_uuid', connection_info={'format': file_format})
    self.mock_object(self.connector.volume_api, 'attachment_get', return_value=fake_attachment)
    if encrypted or file_format == 'qcow2':
        self.assertRaises(exceptions.BackendException, self.connector.connect_volume, fake_vol)
    else:
        fake_hash = 'fake_hash'
        fake_path = {'path': os.path.join(self.mountpath, fake_hash, self.connection_info['name'])}
        self.mock_object(nfs.NfsBrickConnector, 'get_hash_str', return_value=fake_hash)
        fake_dev_path = self.connector.connect_volume(fake_vol)
        nfs.mount.mount.assert_called_once_with('nfs', self.connection_info['export'], self.connection_info['name'], os.path.join(self.mountpath, fake_hash), self.connector.host, self.connector.root_helper, self.connection_info['options'])
        self.assertEqual(fake_path['path'], fake_dev_path['path'])