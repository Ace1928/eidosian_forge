import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_object_properties_with_empty_moref(self):
    vim = mock.Mock()
    ret = vim_util.get_object_properties(vim, None, None)
    self.assertIsNone(ret)