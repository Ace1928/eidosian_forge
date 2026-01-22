import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connectors_list_marker(self):
    volume_connectors = self.mgr.list(marker=CONNECTOR1['uuid'])
    expect = [('GET', '/v1/volume/connectors/?marker=%s' % CONNECTOR1['uuid'], {}, None)]
    expect_connectors = [CONNECTOR2]
    self._validate_list(expect, expect_connectors, volume_connectors)