import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_find_extension(self):
    vim = mock.Mock()
    ret = vim_util.find_extension(vim, 'fake-key')
    self.assertIsNotNone(ret)
    service_content = vim.service_content
    vim.FindExtension.assert_called_once_with(service_content.extensionManager, extensionKey='fake-key')