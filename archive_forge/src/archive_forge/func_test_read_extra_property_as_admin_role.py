from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_read_extra_property_as_admin_role(self):
    extra_properties = {'foo': 'bar', 'ping': 'pong'}
    context = glance.context.RequestContext(roles=['admin'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    test_result = extra_prop_proxy['foo']
    self.assertEqual('bar', test_result)