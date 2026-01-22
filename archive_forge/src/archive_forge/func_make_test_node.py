from heat.engine import node_data
from heat.tests import common
def make_test_node():
    return node_data.NodeData.from_dict(make_test_data())