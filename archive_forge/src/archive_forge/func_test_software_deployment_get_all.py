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
def test_software_deployment_get_all(self):
    self.assertEqual([], db_api.software_deployment_get_all(self.ctx))
    values = self._deployment_values()
    deployment = db_api.software_deployment_create(self.ctx, values)
    self.assertIsNotNone(deployment)
    deployments = db_api.software_deployment_get_all(self.ctx)
    self.assertEqual(1, len(deployments))
    self.assertEqual(deployment.id, deployments[0].id)
    deployments = db_api.software_deployment_get_all(self.ctx, server_id=values['server_id'])
    self.assertEqual(1, len(deployments))
    self.assertEqual(deployment.id, deployments[0].id)
    deployments = db_api.software_deployment_get_all(self.ctx, server_id=str(uuid.uuid4()))
    self.assertEqual([], deployments)
    admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
    deployments = db_api.software_deployment_get_all(admin_ctx)
    self.assertEqual(1, len(deployments))