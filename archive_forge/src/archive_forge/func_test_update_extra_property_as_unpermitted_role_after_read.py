from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_update_extra_property_as_unpermitted_role_after_read(self):
    extra_properties = {'spl_read_prop': 'bar'}
    context = glance.context.RequestContext(roles=['spl_role'])
    extra_prop_proxy = property_protections.ExtraPropertiesProxy(context, extra_properties, self.property_rules)
    self.assertRaises(exception.ReservedProperty, extra_prop_proxy.__setitem__, 'spl_read_prop', 'par')