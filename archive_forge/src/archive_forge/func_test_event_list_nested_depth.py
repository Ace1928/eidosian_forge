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
@tools.stack_context('service_event_list_test_stack')
@mock.patch.object(service.EngineService, '_get_stack')
def test_event_list_nested_depth(self, mock_get):
    mock_get.return_value = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    events = self.eng.list_events(self.ctx, self.stack.identifier(), nested_depth=1)
    self.assertEqual(4, len(events))
    for ev in events:
        self.assertIn('root_stack_id', ev)
    mock_get.assert_called_once_with(self.ctx, self.stack.identifier(), show_deleted=True)