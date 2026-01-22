import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_list_software_configs(self):
    config = self._create_software_config()
    self.assertIsNotNone(config)
    config_id = config['id']
    configs = self.engine.list_software_configs(self.ctx)
    self.assertIsNotNone(configs)
    config_ids = [x['id'] for x in configs]
    self.assertIn(config_id, config_ids)
    admin_cntx = utils.dummy_context(is_admin=True)
    admin_config = self._create_software_config(context=admin_cntx)
    admin_config_id = admin_config['id']
    configs = self.engine.list_software_configs(admin_cntx)
    self.assertIsNotNone(configs)
    config_ids = [x['id'] for x in configs]
    project_ids = [x['project'] for x in configs]
    self.assertEqual(2, len(project_ids))
    self.assertEqual(2, len(config_ids))
    self.assertIn(config_id, config_ids)
    self.assertIn(admin_config_id, config_ids)