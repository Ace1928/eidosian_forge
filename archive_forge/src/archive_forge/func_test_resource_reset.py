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
def test_resource_reset(self):
    resource_progress = create_resource(self.ctx, self.stack, status='IN_PROGRESS', engine_id=UUID2)
    resource_complete = create_resource(self.ctx, self.stack)
    db_api.reset_stack_status(self.ctx, self.stack.id)
    resource_complete = db_api.resource_get(self.ctx, resource_complete.id)
    resource_progress = db_api.resource_get(self.ctx, resource_progress.id)
    self.assertEqual('complete', resource_complete.status)
    self.assertEqual('FAILED', resource_progress.status)
    self.assertIsNone(resource_progress.engine_id)