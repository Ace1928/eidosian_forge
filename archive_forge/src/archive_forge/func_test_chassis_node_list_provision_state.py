import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.chassis
def test_chassis_node_list_provision_state(self):
    nodes = self.mgr.list_nodes(CHASSIS['uuid'], provision_state='available')
    expect = [('GET', '/v1/chassis/%s/nodes?provision_state=available' % CHASSIS['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(nodes))