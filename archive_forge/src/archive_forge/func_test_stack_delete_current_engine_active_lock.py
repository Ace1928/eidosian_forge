from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from oslo_utils import timeutils
from heat.common import exception
from heat.common import service_utils
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(parser.Stack, 'load')
@mock.patch.object(stack_lock.StackLock, 'try_acquire')
@mock.patch.object(stack_lock.StackLock, 'acquire')
@mock.patch.object(timeutils.StopWatch, 'expired')
def test_stack_delete_current_engine_active_lock(self, mock_expired, mock_acquire, mock_try, mock_load):
    cfg.CONF.set_override('error_wait_time', 0)
    self.man.engine_id = service_utils.generate_engine_id()
    stack_name = 'service_delete_test_stack_current_active_lock'
    stack = tools.get_stack(stack_name, self.ctx)
    sid = stack.store()
    stack_lock_object.StackLock.create(self.ctx, stack.id, self.man.engine_id)
    st = stack_object.Stack.get_by_id(self.ctx, sid)
    mock_load.return_value = stack
    mock_try.return_value = self.man.engine_id
    mock_send = self.patchobject(self.man.thread_group_mgr, 'send')
    mock_expired.side_effect = [False, True]
    with mock.patch.object(self.man.thread_group_mgr, 'stop') as mock_stop:
        self.assertIsNone(self.man.delete_stack(self.ctx, stack.identifier()))
        self.man.thread_group_mgr.groups[sid].wait()
        mock_load.assert_called_with(self.ctx, stack=st)
        mock_send.assert_called_once_with(stack.id, 'cancel')
        mock_stop.assert_called_once_with(stack.id)
    self.man.thread_group_mgr.stop(sid, graceful=True)
    self.assertEqual(2, len(mock_load.mock_calls))
    mock_try.assert_called_with()
    mock_acquire.assert_called_once_with(True)