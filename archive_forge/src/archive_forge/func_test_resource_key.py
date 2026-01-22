from heat.engine import node_data
from heat.tests import common
def test_resource_key(self):
    nd = make_test_node()
    self.assertEqual(42, nd.primary_key)