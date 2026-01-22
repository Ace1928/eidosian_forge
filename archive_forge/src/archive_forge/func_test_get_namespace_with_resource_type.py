import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_get_namespace_with_resource_type(self):
    namespace = self.controller.get(NAMESPACE6, resource_type=RESOURCE_TYPE1)
    self.assertEqual(NAMESPACE6, namespace.namespace)
    self.assertTrue(namespace.protected)