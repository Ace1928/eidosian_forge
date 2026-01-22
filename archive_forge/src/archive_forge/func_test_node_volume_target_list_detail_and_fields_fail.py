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
def test_node_volume_target_list_detail_and_fields_fail(self):
    self.assertRaises(exc.InvalidAttribute, self.mgr.list_volume_targets, NODE1['uuid'], detail=True, fields=['uuid', 'extra'])