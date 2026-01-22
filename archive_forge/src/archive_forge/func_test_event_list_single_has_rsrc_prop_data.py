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
@tools.stack_context('service_event_list_single_event')
@mock.patch.object(service.EngineService, '_get_stack')
def test_event_list_single_has_rsrc_prop_data(self, mock_get):
    mock_get.return_value = stack_object.Stack.get_by_id(self.ctx, self.stack.id)
    events = self.eng.list_events(self.ctx, self.stack.identifier())
    self.assertEqual(4, len(events))
    for ev in events:
        self.assertNotIn('resource_properties', ev)
    event_objs = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)
    for i in range(2):
        event_uuid = event_objs[i]['uuid']
        events = self.eng.list_events(self.ctx, self.stack.identifier(), filters={'uuid': event_uuid})
        self.assertEqual(1, len(events))
        self.assertIn('resource_properties', events[0])
        if i > 0:
            self.assertEqual(4, len(events[0]['resource_properties']))