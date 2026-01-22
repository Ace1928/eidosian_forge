import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('tarfile.open')
@mock.patch('oslo_vmware.image_util.get_vmdk_name_from_ovf')
def test_get_vmdk_handle(self, get_vmdk_name_from_ovf, tar_open):
    ovf_info = mock.Mock()
    ovf_info.name = 'test.ovf'
    vmdk_info = mock.Mock()
    vmdk_info.name = 'test.vmdk'
    tar = mock.Mock()
    tar.__iter__ = mock.Mock(return_value=iter([ovf_info, vmdk_info]))
    tar.__enter__ = mock.Mock(return_value=tar)
    tar.__exit__ = mock.Mock(return_value=None)
    tar_open.return_value = tar
    ovf_handle = mock.Mock()
    get_vmdk_name_from_ovf.return_value = 'test.vmdk'
    vmdk_handle = mock.Mock()
    tar.extractfile.side_effect = [ovf_handle, vmdk_handle]
    ova_handle = mock.sentinel.ova_handle
    ret = image_transfer._get_vmdk_handle(ova_handle)
    self.assertEqual(vmdk_handle, ret)
    tar_open.assert_called_once_with(mode='r|', fileobj=ova_handle)
    self.assertEqual([mock.call(ovf_info), mock.call(vmdk_info)], tar.extractfile.call_args_list)
    get_vmdk_name_from_ovf.assert_called_once_with(ovf_handle)