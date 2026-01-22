from unittest import mock
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.engine import worker
from heat.objects import stack as stack_objects
from heat.rpc import worker_client as wc
from heat.tests import common
from heat.tests import utils
@mock.patch.object(worker, '_wait_for_cancellation')
@mock.patch.object(worker, '_cancel_check_resource')
@mock.patch.object(wc.WorkerClient, 'cancel_check_resource')
@mock.patch.object(db_api, 'engine_get_all_locked_by_stack')
def test_cancel_workers_with_resources_found(self, mock_get_locked, mock_ccr, mock_wccr, mock_wc):
    mock_tgm = mock.Mock()
    _worker = worker.WorkerService('host-1', 'topic-1', 'engine-001', mock_tgm)
    stack = mock.MagicMock()
    stack.id = 'stack_id'
    mock_get_locked.return_value = ['engine-001', 'engine-007', 'engine-008']
    worker._cancel_workers(stack, mock_tgm, 'engine-001', _worker._rpc_client)
    mock_wccr.assert_called_once_with(stack.id, 'engine-001', mock_tgm)
    self.assertEqual(2, mock_ccr.call_count)
    calls = [mock.call(stack.context, stack.id, 'engine-007'), mock.call(stack.context, stack.id, 'engine-008')]
    mock_ccr.assert_has_calls(calls, any_order=True)
    self.assertTrue(mock_wc.called)