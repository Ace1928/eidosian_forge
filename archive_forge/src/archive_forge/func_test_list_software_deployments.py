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
def test_list_software_deployments(self):
    stack_name = 'test_list_software_deployments'
    t = template_format.parse(tools.wp_template)
    stack = utils.parse_stack(t, stack_name=stack_name)
    fc = tools.setup_mocks_with_mock(self, stack)
    stack.store()
    stack.create()
    server = stack['WebServer']
    server_id = server.resource_id
    deployment = self._create_software_deployment(server_id=server_id)
    deployment_id = deployment['id']
    self.assertIsNotNone(deployment)
    deployments = self.engine.list_software_deployments(self.ctx, server_id=None)
    self.assertIsNotNone(deployments)
    deployment_ids = [x['id'] for x in deployments]
    self.assertIn(deployment_id, deployment_ids)
    self.assertIn(deployment, deployments)
    deployments = self.engine.list_software_deployments(self.ctx, server_id=str(uuid.uuid4()))
    self.assertEqual([], deployments)
    deployments = self.engine.list_software_deployments(self.ctx, server_id=server.resource_id)
    self.assertEqual([deployment], deployments)
    rsrcs = resource_objects.Resource.get_all_by_physical_resource_id(self.ctx, server_id)
    self.assertEqual(deployment['config_id'], rsrcs[0].rsrc_metadata.get('deployments')[0]['id'])
    tools.validate_setup_mocks_with_mock(stack, fc)