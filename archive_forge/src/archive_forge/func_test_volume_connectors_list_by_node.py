import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connectors_list_by_node(self):
    volume_connectors = self.mgr.list(node=NODE_UUID)
    expect = [('GET', '/v1/volume/connectors/?node=%s' % NODE_UUID, {}, None)]
    expect_connectors = [CONNECTOR1]
    self._validate_list(expect, expect_connectors, volume_connectors)