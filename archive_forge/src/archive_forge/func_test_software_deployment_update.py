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
def test_software_deployment_update(self):
    deployment_id = str(uuid.uuid4())
    err = self.assertRaises(exception.NotFound, db_api.software_deployment_update, self.ctx, deployment_id, values={})
    self.assertIn(deployment_id, str(err))
    values = self._deployment_values()
    deployment = db_api.software_deployment_create(self.ctx, values)
    deployment_id = deployment.id
    values = {'status': 'COMPLETED'}
    deployment = db_api.software_deployment_update(self.ctx, deployment_id, values)
    self.assertIsNotNone(deployment)
    self.assertEqual(values['status'], deployment.status)
    admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
    values = {'status': 'FAILED'}
    deployment = db_api.software_deployment_update(admin_ctx, deployment_id, values)
    self.assertIsNotNone(deployment)
    self.assertEqual(values['status'], deployment.status)