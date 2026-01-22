import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.build_recursive_traversal_spec')
def test_get_objects(self, build_recursive_traversal_spec):
    vim = mock.Mock()
    trav_spec = mock.Mock()
    build_recursive_traversal_spec.return_value = trav_spec
    max_objects = 10
    _type = 'VirtualMachine'

    def vim_RetrievePropertiesEx_side_effect(pc, specSet, options):
        self.assertTrue(pc is vim.service_content.propertyCollector)
        self.assertEqual(max_objects, options.maxObjects)
        self.assertEqual(1, len(specSet))
        property_filter_spec = specSet[0]
        propSet = property_filter_spec.propSet
        self.assertEqual(1, len(propSet))
        prop_spec = propSet[0]
        self.assertFalse(prop_spec.all)
        self.assertEqual(['name'], prop_spec.pathSet)
        self.assertEqual(_type, prop_spec.type)
        objSet = property_filter_spec.objectSet
        self.assertEqual(1, len(objSet))
        obj_spec = objSet[0]
        self.assertTrue(obj_spec.obj is vim.service_content.rootFolder)
        self.assertEqual([trav_spec], obj_spec.selectSet)
        self.assertFalse(obj_spec.skip)
    vim.RetrievePropertiesEx.side_effect = vim_RetrievePropertiesEx_side_effect
    vim_util.get_objects(vim, _type, max_objects)
    self.assertEqual(1, vim.RetrievePropertiesEx.call_count)