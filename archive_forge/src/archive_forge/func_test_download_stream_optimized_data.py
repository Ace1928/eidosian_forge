import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('oslo_vmware.rw_handles.VmdkWriteHandle')
@mock.patch.object(image_transfer, '_start_transfer')
def test_download_stream_optimized_data(self, fake_transfer, fake_rw_handles_VmdkWriteHandle):
    context = mock.Mock()
    session = mock.Mock()
    read_handle = mock.Mock()
    timeout_secs = 10
    image_size = 1000
    host = '127.0.0.1'
    port = 443
    resource_pool = 'rp-1'
    vm_folder = 'folder-1'
    vm_import_spec = None
    fake_VmdkWriteHandle = mock.Mock()
    fake_VmdkWriteHandle.get_imported_vm = mock.Mock()
    fake_rw_handles_VmdkWriteHandle.return_value = fake_VmdkWriteHandle
    image_transfer.download_stream_optimized_data(context, timeout_secs, read_handle, session=session, host=host, port=port, resource_pool=resource_pool, vm_folder=vm_folder, vm_import_spec=vm_import_spec, image_size=image_size)
    fake_rw_handles_VmdkWriteHandle.assert_called_once_with(session, host, port, resource_pool, vm_folder, vm_import_spec, image_size, 'PUT')
    fake_transfer.assert_called_once_with(read_handle, fake_VmdkWriteHandle, timeout_secs)
    fake_VmdkWriteHandle.get_imported_vm.assert_called_once_with()