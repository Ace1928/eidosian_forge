from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_node_data_ok(self):
    self.resource.action = self.resource.CREATE
    expected_input_data = {'attrs': {(u'flat_dict', u'key2'): 'val2', (u'flat_dict', u'key3'): 'val3', (u'nested_dict', u'dict', u'a'): 1, (u'nested_dict', u'dict', u'b'): 2}, 'id': mock.ANY, 'reference_id': 'A', 'name': 'A', 'uuid': mock.ANY, 'action': mock.ANY, 'status': mock.ANY}
    actual_input_data = self.resource.node_data()
    self.assertEqual(expected_input_data, actual_input_data.as_dict())