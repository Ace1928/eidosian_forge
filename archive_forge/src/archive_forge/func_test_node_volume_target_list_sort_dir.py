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
def test_node_volume_target_list_sort_dir(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = node.NodeManager(self.api)
    volume_targets = self.mgr.list_volume_targets(NODE1['uuid'], sort_dir='desc')
    expect = [('GET', '/v1/nodes/%s/volume/targets?sort_dir=desc' % NODE1['uuid'], {}, None)]
    self._validate_node_volume_target_list(expect, volume_targets)