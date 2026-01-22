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
@mock.patch.object(service_utils, 'engine_alive')
@mock.patch.object(timeutils.StopWatch, 'expired')
def test_stack_delete_other_engine_active_lock_failed(self, mock_expired, mock_alive, mock_try, mock_load):
    cfg.CONF.set_override('error_wait_time', 0)
    OTHER_ENGINE = 'other-engine-fake-uuid'
    self.man.engine_id = service_utils.generate_engine_id()
    self.man.listener = service.EngineListener(self.man.host, self.man.engine_id, self.man.thread_group_mgr)
    stack_name = 'service_delete_test_stack_other_engine_lock_fail'
    stack = tools.get_stack(stack_name, self.ctx)
    sid = stack.store()
    stack_lock_object.StackLock.create(self.ctx, stack.id, OTHER_ENGINE)
    st = stack_object.Stack.get_by_id(self.ctx, sid)
    mock_load.return_value = stack
    mock_try.return_value = OTHER_ENGINE
    mock_alive.return_value = True
    mock_expired.side_effect = [False, True]
    mock_call = self.patchobject(self.man, '_remote_call', return_value=False)
    ex = self.assertRaises(dispatcher.ExpectedException, self.man.delete_stack, self.ctx, stack.identifier())
    self.assertEqual(exception.EventSendFailed, ex.exc_info[0])
    mock_load.assert_called_once_with(self.ctx, stack=st)
    mock_try.assert_called_once_with()
    mock_alive.assert_called_once_with(self.ctx, OTHER_ENGINE)
    mock_call.assert_called_once_with(self.ctx, OTHER_ENGINE, mock.ANY, 'send', message='cancel', stack_identity=mock.ANY)