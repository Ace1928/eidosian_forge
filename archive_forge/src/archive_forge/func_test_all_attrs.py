from heat.engine import node_data
from heat.tests import common
def test_all_attrs(self):
    nd = make_test_node()
    self.assertEqual({'foo': 'bar'}, nd.attributes())