import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_detail(self):
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], detail=True)
    expect = [('GET', '/v1/chassis/%s/nodes/detail' % CHASSIS['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(nodes))
    self.assertEqual(NODE['uuid'], nodes[0].uuid)