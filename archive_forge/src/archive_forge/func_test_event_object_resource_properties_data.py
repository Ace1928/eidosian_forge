from unittest import mock
from oslo_config import cfg
import uuid
from heat.db import api as db_api
from heat.db import models
from heat.engine import event
from heat.engine import stack
from heat.engine import template
from heat.objects import event as event_object
from heat.objects import resource_properties_data as rpd_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def test_event_object_resource_properties_data(self):
    cfg.CONF.set_override('encrypt_parameters_and_properties', True)
    data = {'p1': 'hello', 'p2': 'too soon?'}
    rpd_obj = rpd_object.ResourcePropertiesData().create_or_update(self.ctx, data)
    e_obj = event_object.Event().create(self.ctx, {'stack_id': self.stack.id, 'uuid': str(uuid.uuid4()), 'rsrc_prop_data_id': rpd_obj.id})
    e_obj = event_object.Event.get_all_by_stack(utils.dummy_context(), self.stack.id)[0]
    self.assertEqual(data, e_obj.resource_properties)