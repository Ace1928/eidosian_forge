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
def test_event_create(self):
    stack = create_stack(self.ctx, self.template, self.user_creds)
    event = create_event(self.ctx, stack_id=stack.id)
    with db_api.context_manager.reader.using(self.ctx):
        ret_event = self.ctx.session.query(models.Event).filter_by(id=event.id).options(orm.joinedload(models.Event.rsrc_prop_data)).first()
    self.assertIsNotNone(ret_event)
    self.assertEqual(stack.id, ret_event.stack_id)
    self.assertEqual('create', ret_event.resource_action)
    self.assertEqual('complete', ret_event.resource_status)
    self.assertEqual('res', ret_event.resource_name)
    self.assertEqual(UUID1, ret_event.physical_resource_id)
    self.assertEqual('create_complete', ret_event.resource_status_reason)
    self.assertEqual({'foo2': 'ev_bar'}, ret_event.rsrc_prop_data.data)