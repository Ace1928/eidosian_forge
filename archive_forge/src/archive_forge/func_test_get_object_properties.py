import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.cancel_retrieval')
def test_get_object_properties(self, cancel_retrieval):
    vim = mock.Mock()
    moref = vim_util.get_moref('fake-ref', 'VirtualMachine')
    retrieve_result = mock.Mock()

    def vim_RetrievePropertiesEx_side_effect(pc, specSet, options, skip_op_id=False):
        self.assertTrue(pc is vim.service_content.propertyCollector)
        self.assertEqual(1, options.maxObjects)
        self.assertEqual(1, len(specSet))
        property_filter_spec = specSet[0]
        propSet = property_filter_spec.propSet
        self.assertEqual(1, len(propSet))
        prop_spec = propSet[0]
        self.assertTrue(prop_spec.all)
        self.assertEqual(['name'], prop_spec.pathSet)
        self.assertEqual(vim_util.get_moref_type(moref), prop_spec.type)
        objSet = property_filter_spec.objectSet
        self.assertEqual(1, len(objSet))
        obj_spec = objSet[0]
        self.assertEqual(moref, obj_spec.obj)
        self.assertEqual([], obj_spec.selectSet)
        self.assertFalse(obj_spec.skip)
        return retrieve_result
    vim.RetrievePropertiesEx.side_effect = vim_RetrievePropertiesEx_side_effect
    res = vim_util.get_object_properties(vim, moref, None)
    self.assertEqual(1, vim.RetrievePropertiesEx.call_count)
    self.assertTrue(res is retrieve_result.objects)
    cancel_retrieval.assert_called_once_with(vim, retrieve_result)