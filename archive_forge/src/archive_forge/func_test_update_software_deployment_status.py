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
def test_update_software_deployment_status(self):
    server_id = str(uuid.uuid4())
    mock_push = self.patchobject(self.engine.software_config, '_push_metadata_software_deployments')
    deployment = self._create_software_deployment(server_id=server_id)
    self.assertIsNotNone(deployment)
    deployment_id = deployment['id']
    deployment_action = deployment['action']
    self.assertEqual('INIT', deployment_action)
    updated = self.engine.update_software_deployment(self.ctx, deployment_id=deployment_id, config_id=None, input_values=None, output_values={}, action='DEPLOY', status='WAITING', status_reason='', updated_at=None)
    self.assertIsNotNone(updated)
    self.assertEqual('DEPLOY', updated['action'])
    self.assertEqual('WAITING', updated['status'])
    mock_push.assert_called_once_with(self.ctx, server_id, None)