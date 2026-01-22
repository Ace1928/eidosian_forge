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
def test_stack_set_status_failed_release_lock(self):
    stack = create_stack(self.ctx, self.template, self.user_creds)
    values = {'name': 'db_test_stack_name2', 'action': 'update', 'status': 'failed', 'status_reason': 'update_failed', 'timeout': '90', 'current_traversal': 'another-dummy-uuid'}
    db_api.stack_lock_create(self.ctx, stack.id, UUID1)
    observed = db_api.persist_state_and_release_lock(self.ctx, UUID2, UUID1, values)
    self.assertTrue(observed)