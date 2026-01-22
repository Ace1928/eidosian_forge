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
def test_resource_get_all_by_stack(self):
    stack1 = create_stack(self.ctx, self.template, self.user_creds)
    stack2 = create_stack(self.ctx, self.template, self.user_creds)
    values = [{'name': 'res1', 'stack_id': self.stack.id}, {'name': 'res2', 'stack_id': self.stack.id}, {'name': 'res3', 'stack_id': self.stack.id}, {'name': 'res4', 'stack_id': stack1.id}]
    [create_resource(self.ctx, self.stack, False, **val) for val in values]
    resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id)
    self.assertEqual(3, len(resources))
    self.assertEqual('res1', resources.get('res1').name)
    self.assertEqual('res2', resources.get('res2').name)
    self.assertEqual('res3', resources.get('res3').name)
    resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id, filters=dict(name='res1'))
    self.assertEqual(1, len(resources))
    self.assertEqual('res1', resources.get('res1').name)
    resources = db_api.resource_get_all_by_stack(self.ctx, self.stack.id, filters=dict(name=['res1', 'res2']))
    self.assertEqual(2, len(resources))
    self.assertEqual('res1', resources.get('res1').name)
    self.assertEqual('res2', resources.get('res2').name)
    self.assertEqual({}, db_api.resource_get_all_by_stack(self.ctx, stack2.id))