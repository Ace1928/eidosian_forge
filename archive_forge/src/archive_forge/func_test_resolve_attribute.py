from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine import attributes
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine.resources.openstack.neutron import neutron as nr
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_resolve_attribute(self):
    res = self._get_some_neutron_resource()
    res.attributes_schema.update({'attr2': attributes.Schema(type=attributes.Schema.STRING)})
    res.attributes = attributes.Attributes(res.name, res.attributes_schema, res._resolve_any_attribute)
    side_effect = [{'attr1': 'val1', 'attr2': 'val2'}, {'attr1': 'val1', 'attr2': 'val2'}, {'attr1': 'val1', 'attr2': 'val2'}, qe.NotFound]
    self.patchobject(res, '_show_resource', side_effect=side_effect)
    res.resource_id = 'resource_id'
    self.assertEqual({'attr1': 'val1', 'attr2': 'val2'}, res.FnGetAtt('show'))
    self.assertEqual('val2', res.attributes['attr2'])
    self.assertRaises(KeyError, res._resolve_any_attribute, 'attr3')
    self.assertIsNone(res._resolve_any_attribute('attr1'))
    res.resource_id = None
    self.assertEqual('val2', res.FnGetAtt('attr2'))
    self.assertIsNone(res.FnGetAtt('show'))
    res.attributes.reset_resolved_values()
    self.assertIsNone(res.FnGetAtt('attr2'))