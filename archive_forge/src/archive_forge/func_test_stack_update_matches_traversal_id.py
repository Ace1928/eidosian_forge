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
def test_stack_update_matches_traversal_id(self):
    stack = create_stack(self.ctx, self.template, self.user_creds)
    values = {'current_traversal': 'another-dummy-uuid'}
    updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl='dummy-uuid')
    self.assertTrue(updated)
    stack = db_api.stack_get(self.ctx, stack.id)
    self.assertEqual('another-dummy-uuid', stack.current_traversal)
    matching_uuid = 'another-dummy-uuid'
    updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl=matching_uuid)
    self.assertTrue(updated)
    diff_uuid = 'some-other-dummy-uuid'
    updated = db_api.stack_update(self.ctx, stack.id, values, exp_trvsl=diff_uuid)
    self.assertFalse(updated)