import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_update_progress(self):
    chunk_size = len('fake-data')
    vmdk_size = chunk_size * 10
    session = self._create_mock_session(True, 10)
    handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', vmdk_size)
    data = handle.read(chunk_size)
    handle.update_progress()
    self.assertEqual('fake-data', data)