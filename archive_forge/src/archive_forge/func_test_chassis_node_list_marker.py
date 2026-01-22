import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_marker(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = ironicclient.v1.chassis.ChassisManager(self.api)
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], marker=NODE['uuid'])
    expect = [('GET', '/v1/chassis/%s/nodes?marker=%s' % (CHASSIS['uuid'], NODE['uuid']), {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(nodes, HasLength(1))
    self.assertEqual(NODE['uuid'], nodes[0].uuid)