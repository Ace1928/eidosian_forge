import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_build_object_spec(self):
    client_factory = mock.Mock()
    root_folder = mock.Mock()
    specs = [mock.Mock()]
    obj_spec = vim_util.build_object_spec(client_factory, root_folder, specs)
    self.assertEqual(root_folder, obj_spec.obj)
    self.assertEqual(specs, obj_spec.selectSet)
    self.assertFalse(obj_spec.skip)