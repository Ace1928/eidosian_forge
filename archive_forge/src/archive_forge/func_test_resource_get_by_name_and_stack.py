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
def test_resource_get_by_name_and_stack(self):
    create_resource(self.ctx, self.stack)
    ret_res = db_api.resource_get_by_name_and_stack(self.ctx, 'test_resource_name', self.stack.id)
    self.assertIsNotNone(ret_res)
    self.assertEqual('test_resource_name', ret_res.name)
    self.assertEqual(self.stack.id, ret_res.stack_id)
    self.assertIsNone(db_api.resource_get_by_name_and_stack(self.ctx, 'abc', self.stack.id))