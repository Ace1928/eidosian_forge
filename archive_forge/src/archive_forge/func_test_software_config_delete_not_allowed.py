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
def test_software_config_delete_not_allowed(self):
    tenant_id = self.ctx.tenant_id
    config = db_api.software_config_create(self.ctx, {'name': 'config_mysql', 'tenant': tenant_id})
    config_id = config.id
    values = {'tenant': tenant_id, 'stack_user_project_id': str(uuid.uuid4()), 'config_id': config_id, 'server_id': str(uuid.uuid4())}
    db_api.software_deployment_create(self.ctx, values)
    err = self.assertRaises(exception.InvalidRestrictedAction, db_api.software_config_delete, self.ctx, config_id)
    msg = 'Software config with id %s can not be deleted as it is referenced' % config_id
    self.assertIn(msg, str(err))