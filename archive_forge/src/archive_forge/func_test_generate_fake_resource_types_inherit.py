from openstack import format as _format
from openstack import resource
from openstack.test import fakes
from openstack.tests.unit import base
def test_generate_fake_resource_types_inherit(self):

    class Fake(resource.Resource):
        a = resource.Body('a', type=str)

    class FakeInherit(resource.Resource):
        a = resource.Body('a', type=Fake)
    res = fakes.generate_fake_resource(FakeInherit)
    self.assertIsInstance(res.a, Fake)
    self.assertIsInstance(res.a.a, str)