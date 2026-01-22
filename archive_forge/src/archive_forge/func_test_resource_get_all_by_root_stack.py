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
def test_resource_get_all_by_root_stack(self):
    stack1 = create_stack(self.ctx, self.template, self.user_creds)
    stack2 = create_stack(self.ctx, self.template, self.user_creds)
    create_resource(self.ctx, self.stack, name='res1', root_stack_id=self.stack.id)
    create_resource(self.ctx, self.stack, name='res2', root_stack_id=self.stack.id)
    create_resource(self.ctx, self.stack, name='res3', root_stack_id=self.stack.id)
    create_resource(self.ctx, stack1, name='res4', root_stack_id=self.stack.id)
    resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id)
    self.assertEqual(4, len(resources))
    resource_names = [r.name for r in resources.values()]
    self.assertEqual(['res1', 'res2', 'res3', 'res4'], sorted(resource_names))
    resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id, filters=dict(name='res1'))
    self.assertEqual(1, len(resources))
    resource_names = [r.name for r in resources.values()]
    self.assertEqual(['res1'], resource_names)
    self.assertEqual(1, len(resources))
    resources = db_api.resource_get_all_by_root_stack(self.ctx, self.stack.id, filters=dict(name=['res1', 'res2']))
    self.assertEqual(2, len(resources))
    resource_names = [r.name for r in resources.values()]
    self.assertEqual(['res1', 'res2'], sorted(resource_names))
    self.assertEqual({}, db_api.resource_get_all_by_root_stack(self.ctx, stack2.id))