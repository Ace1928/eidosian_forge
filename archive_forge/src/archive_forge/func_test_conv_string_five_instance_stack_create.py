from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_conv_string_five_instance_stack_create(self, mock_cr):
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.store()
    stack.converge_stack(template=stack.t, action=stack.CREATE)
    self.assertIsNone(stack.ext_rsrcs_db)
    self.assertEqual([((1, True), (3, True)), ((2, True), (3, True)), ((3, True), (4, True)), ((3, True), (5, True))], sorted(stack.convergence_dependencies._graph.edges()))
    stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
    self.assertIsNotNone(stack_db.current_traversal)
    self.assertIsNotNone(stack_db.raw_template_id)
    self.assertIsNone(stack_db.prev_raw_template_id)
    self.assertTrue(stack_db.convergence)
    self.assertEqual(sorted([[[3, True], [5, True]], [[3, True], [4, True]], [[1, True], [3, True]], [[2, True], [3, True]]]), sorted(stack_db.current_deps['edges']))
    for entity_id in [5, 4, 3, 2, 1, stack_db.id]:
        sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, True)
        self.assertIsNotNone(sync_point)
        self.assertEqual(stack_db.id, sync_point.stack_id)
    leaves = set(stack.convergence_dependencies.leaves())
    expected_calls = []
    for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
        expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
    self.assertEqual(expected_calls, mock_cr.mock_calls)