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
def test_validate_properties(self):
    vs = {'router:external': True}
    data = {'admin_state_up': False, 'value_specs': vs}
    p = properties.Properties(net.Net.properties_schema, data)
    self.assertIsNone(nr.NeutronResource.validate_properties(p))
    vs['foo'] = '1234'
    self.assertIsNone(nr.NeutronResource.validate_properties(p))
    vs.pop('foo')
    banned_keys = {'shared': True, 'name': 'foo', 'tenant_id': '1234'}
    for key, val in banned_keys.items():
        vs.update({key: val})
        msg = '%s not allowed in value_specs' % key
        self.assertEqual(msg, nr.NeutronResource.validate_properties(p))
        vs.pop(key)