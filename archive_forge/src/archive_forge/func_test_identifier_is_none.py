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
def test_identifier_is_none(self):
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'wibble', None, None, self.resource.name, self.resource.type())
    self.assertIsNone(e.identifier())
    e.store()
    self.assertIsNotNone(e.identifier())