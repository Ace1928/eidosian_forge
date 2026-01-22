import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
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