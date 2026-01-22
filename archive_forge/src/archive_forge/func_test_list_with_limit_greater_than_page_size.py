import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_with_limit_greater_than_page_size(self):
    namespaces = self.controller.list(page_size=1, limit=2)
    self.assertEqual(2, len(namespaces))
    self.assertEqual(NAMESPACE7, namespaces[0]['namespace'])
    self.assertEqual(NAMESPACE8, namespaces[1]['namespace'])