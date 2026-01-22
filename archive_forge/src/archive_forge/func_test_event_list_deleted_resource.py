from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from heat.common import context
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import event as event_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@tools.stack_context('service_event_list_deleted_resource')
@mock.patch.object(instances.Instance, 'handle_delete')
def test_event_list_deleted_resource(self, mock_delete):
    self.useFixture(conffixture.ConfFixture(cfg.CONF))
    mock_delete.return_value = None
    res._register_class('GenericResourceType', generic_rsrc.GenericResource)
    thread = mock.Mock()
    thread.link = mock.Mock(return_value=None)

    def run(stack_id, func, *args, **kwargs):
        func(*args, **kwargs)
        return thread
    self.eng.thread_group_mgr.start = run
    new_tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'AResource': {'Type': 'GenericResourceType'}}}
    result = self.eng.update_stack(self.ctx, self.stack.identifier(), new_tmpl, None, None, {})
    self.stack = parser.Stack.load(self.ctx, stack_id=result['stack_id'])
    self.assertEqual(result, self.stack.identifier())
    self.assertIsInstance(result, dict)
    self.assertTrue(result['stack_id'])
    events = self.eng.list_events(self.ctx, self.stack.identifier())
    self.assertEqual(10, len(events))
    for ev in events:
        self.assertIn('event_identity', ev)
        self.assertIsInstance(ev['event_identity'], dict)
        self.assertTrue(ev['event_identity']['path'].rsplit('/', 1)[1])
        self.assertIn('resource_name', ev)
        self.assertIn('physical_resource_id', ev)
        self.assertIn('resource_status_reason', ev)
        self.assertIn(ev['resource_action'], ('CREATE', 'UPDATE', 'DELETE'))
        self.assertIn(ev['resource_status'], ('IN_PROGRESS', 'COMPLETE'))
        self.assertIn('resource_type', ev)
        self.assertIn(ev['resource_type'], ('AWS::EC2::Instance', 'GenericResourceType', 'OS::Heat::Stack'))
        self.assertIn('stack_identity', ev)
        self.assertIn('stack_name', ev)
        self.assertEqual(self.stack.name, ev['stack_name'])
        self.assertIn('event_time', ev)
    mock_delete.assert_called_once_with()
    expected = [mock.call(mock.ANY), mock.call(mock.ANY, self.stack.id, mock.ANY)]
    self.assertEqual(expected, thread.link.call_args_list)