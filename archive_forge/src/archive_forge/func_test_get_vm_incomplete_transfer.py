import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_vm_incomplete_transfer(self):
    session = self._create_mock_session()
    handle = rw_handles.VmdkWriteHandle(session, '10.1.2.3', 443, 'rp-1', 'folder-1', None, 100)
    handle._get_progress = mock.Mock(return_value=99)
    session.invoke_api = mock.Mock()
    self.assertRaises(exceptions.ImageTransferException, handle.get_imported_vm)