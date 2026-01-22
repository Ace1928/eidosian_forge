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
def test_node_remove_trait(self):
    trait = 'CUSTOM_FOO'
    resp = self.mgr.remove_trait(NODE1['uuid'], trait)
    expect = [('DELETE', '/v1/nodes/%s/traits/%s' % (NODE1['uuid'], trait), {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(resp)