from glance.api import policy
from glance.api import property_protections
from glance.common import exception
from glance.common import property_utils
import glance.domain
from glance.tests import utils
def test_create_image_no_extra_prop(self):
    self.context = glance.context.RequestContext(tenant=TENANT1, roles=['spl_role'])
    self.image_factory = property_protections.ProtectedImageFactoryProxy(self.factory, self.context, self.property_rules)
    extra_props = {}
    image = self.image_factory.new_image(extra_properties=extra_props)
    expected_extra_props = {}
    self.assertEqual(expected_extra_props, image.extra_properties)