import io
from unittest import mock
from oslo_vmware import exceptions
from oslo_vmware import image_transfer
from oslo_vmware.tests import base
@mock.patch('tarfile.open')
def test_get_vmdk_handle_with_invalid_ova(self, tar_open):
    tar = mock.Mock()
    tar.__iter__ = mock.Mock(return_value=iter([]))
    tar.__enter__ = mock.Mock(return_value=tar)
    tar.__exit__ = mock.Mock(return_value=None)
    tar_open.return_value = tar
    ova_handle = mock.sentinel.ova_handle
    ret = image_transfer._get_vmdk_handle(ova_handle)
    self.assertIsNone(ret)
    tar_open.assert_called_once_with(mode='r|', fileobj=ova_handle)
    self.assertFalse(tar.extractfile.called)