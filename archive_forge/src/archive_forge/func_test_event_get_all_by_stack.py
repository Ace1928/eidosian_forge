import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_event_get_all_by_stack(self):
    stack1 = create_stack(self.ctx, self.template, self.user_creds)
    stack2 = create_stack(self.ctx, self.template, self.user_creds)
    values = [{'stack_id': stack1.id, 'resource_name': 'res1'}, {'stack_id': stack1.id, 'resource_name': 'res2'}, {'stack_id': stack2.id, 'resource_name': 'res3'}]
    [create_event(self.ctx, **val) for val in values]
    self.ctx.project_id = 'tenant1'
    events = db_api.event_get_all_by_stack(self.ctx, stack1.id)
    self.assertEqual(2, len(events))
    self.ctx.project_id = 'tenant2'
    events = db_api.event_get_all_by_stack(self.ctx, stack2.id)
    self.assertEqual(1, len(events))