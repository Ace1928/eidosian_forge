from heat.engine import node_data
from heat.tests import common
def test_resource_name(self):
    nd = make_test_node()
    self.assertEqual('foo', nd.name)