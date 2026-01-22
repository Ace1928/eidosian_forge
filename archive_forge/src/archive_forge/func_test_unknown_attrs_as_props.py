from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_unknown_attrs_as_props(self):

    class Fake(resource.Resource):
        properties = resource.Body('properties')
        _store_unknown_attrs_as_properties = True
    res = fakes.generate_fake_resource(Fake)
    self.assertIsInstance(res.properties, dict)