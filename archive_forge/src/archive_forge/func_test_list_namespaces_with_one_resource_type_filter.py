import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_namespaces_with_one_resource_type_filter(self):
    namespaces = self.controller.list(filters={'resource_types': [RESOURCE_TYPE1]})
    self.assertEqual(1, len(namespaces))
    self.assertEqual(NAMESPACE3, namespaces[0]['namespace'])