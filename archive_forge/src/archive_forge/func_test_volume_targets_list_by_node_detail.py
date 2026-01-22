import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_targets_list_by_node_detail(self):
    volume_targets = self.mgr.list(node=NODE_UUID, detail=True)
    expect = [('GET', '/v1/volume/targets/?detail=True&node=%s' % NODE_UUID, {}, None)]
    expect_targets = [TARGET1]
    self._validate_list(expect, expect_targets, volume_targets)