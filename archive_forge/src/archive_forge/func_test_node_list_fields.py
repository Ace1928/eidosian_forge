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
def test_node_list_fields(self):
    nodes = self.mgr.list(fields=['uuid', 'extra'])
    expect = [('GET', '/v1/nodes/?fields=uuid,extra', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(nodes))