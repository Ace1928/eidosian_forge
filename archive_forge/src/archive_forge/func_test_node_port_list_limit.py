import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def test_node_port_list_limit(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = node.NodeManager(self.api)
    ports = self.mgr.list_ports(NODE1['uuid'], limit=1)
    expect = [('GET', '/v1/nodes/%s/ports?limit=1' % NODE1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(ports, HasLength(1))
    self.assertEqual(PORT['uuid'], ports[0].uuid)
    self.assertEqual(PORT['address'], ports[0].address)