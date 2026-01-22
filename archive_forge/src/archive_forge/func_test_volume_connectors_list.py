import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connectors_list(self):
    volume_connectors = self.mgr.list()
    expect = [('GET', '/v1/volume/connectors', {}, None)]
    expect_connectors = [CONNECTOR1]
    self._validate_list(expect, expect_connectors, volume_connectors)