from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
@mock.patch.object(parser.Stack, '_converge_create_or_update')
@mock.patch.object(parser.Stack, '_send_notification_and_add_event')
def test_convergence_stack_adopt_with_params(self, mock_converge, mock_send_notif):
    cfg.CONF.set_override('enable_stack_adopt', True)
    cfg.CONF.set_override('convergence_engine', True)
    env = {'parameters': {'app_dbx': 'test'}}
    template, adopt_data = self._get_adopt_data_and_template(env)
    result = self._do_adopt('test_adopt_with_params', template, {}, adopt_data)
    stack = stack_object.Stack.get_by_id(self.ctx, result['stack_id'])
    self.assertEqual(template, stack.raw_template.template)
    self.assertEqual(env['parameters'], stack.raw_template.environment['parameters'])
    self.assertTrue(mock_converge.called)