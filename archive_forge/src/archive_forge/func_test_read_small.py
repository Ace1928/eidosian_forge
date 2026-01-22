import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_read_small(self):
    read_data = 'fake'
    session = self._create_mock_session(read_data=read_data)
    read_size = len(read_data)
    handle = rw_handles.VmdkReadHandle(session, '10.1.2.3', 443, 'vm-1', '[ds] disk1.vmdk', read_size * 10)
    handle.read(read_size)
    self.assertEqual(read_size, handle._bytes_read)