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
@mock.patch.object(service_software_config.SoftwareConfigService, '_push_metadata_software_deployments')
def test_signal_software_deployment(self, pmsd):
    self.assertRaises(ValueError, self.engine.signal_software_deployment, self.ctx, None, {}, None)
    deployment_id = str(uuid.uuid4())
    ex = self.assertRaises(dispatcher.ExpectedException, self.engine.signal_software_deployment, self.ctx, deployment_id, {}, None)
    self.assertEqual(exception.NotFound, ex.exc_info[0])
    deployment = self._create_software_deployment()
    deployment_id = deployment['id']
    self.assertIsNone(self.engine.signal_software_deployment(self.ctx, deployment_id, {}, None))
    deployment = self._create_software_deployment(action='INIT', status='IN_PROGRESS')
    deployment_id = deployment['id']
    res = self.engine.signal_software_deployment(self.ctx, deployment_id, {}, None)
    self.assertEqual('deployment %s succeeded' % deployment_id, res)
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.assertEqual('COMPLETE', sd.status)
    self.assertEqual('Outputs received', sd.status_reason)
    self.assertEqual({'deploy_status_code': None, 'deploy_stderr': None, 'deploy_stdout': None}, sd.output_values)
    self.assertIsNotNone(sd.updated_at)
    config = self._create_software_config(outputs=[{'name': 'foo'}])
    deployment = self._create_software_deployment(config_id=config['id'], action='INIT', status='IN_PROGRESS')
    deployment_id = deployment['id']
    result = self.engine.signal_software_deployment(self.ctx, deployment_id, {'foo': 'bar', 'deploy_status_code': 0}, None)
    self.assertEqual('deployment %s succeeded' % deployment_id, result)
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.assertEqual('COMPLETE', sd.status)
    self.assertEqual('Outputs received', sd.status_reason)
    self.assertEqual({'deploy_status_code': 0, 'foo': 'bar', 'deploy_stderr': None, 'deploy_stdout': None}, sd.output_values)
    self.assertIsNotNone(sd.updated_at)
    config = self._create_software_config(outputs=[{'name': 'foo'}])
    deployment = self._create_software_deployment(config_id=config['id'], action='INIT', status='IN_PROGRESS')
    deployment_id = deployment['id']
    result = self.engine.signal_software_deployment(self.ctx, deployment_id, {'foo': 'bar', 'deploy_status_code': -1, 'deploy_stderr': 'Its gone Pete Tong'}, None)
    self.assertEqual('deployment %s failed (-1)' % deployment_id, result)
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.assertEqual('FAILED', sd.status)
    self.assert_status_reason('deploy_status_code : Deployment exited with non-zero status code: -1', sd.status_reason)
    self.assertEqual({'deploy_status_code': -1, 'foo': 'bar', 'deploy_stderr': 'Its gone Pete Tong', 'deploy_stdout': None}, sd.output_values)
    self.assertIsNotNone(sd.updated_at)
    config = self._create_software_config(outputs=[{'name': 'foo', 'error_output': True}])
    deployment = self._create_software_deployment(config_id=config['id'], action='INIT', status='IN_PROGRESS')
    deployment_id = deployment['id']
    result = self.engine.signal_software_deployment(self.ctx, deployment_id, {'foo': 'bar', 'deploy_status_code': -1, 'deploy_stderr': 'Its gone Pete Tong'}, None)
    self.assertEqual('deployment %s failed' % deployment_id, result)
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    self.assertEqual('FAILED', sd.status)
    self.assert_status_reason('foo : bar, deploy_status_code : Deployment exited with non-zero status code: -1', sd.status_reason)
    self.assertEqual({'deploy_status_code': -1, 'foo': 'bar', 'deploy_stderr': 'Its gone Pete Tong', 'deploy_stdout': None}, sd.output_values)
    self.assertIsNotNone(sd.updated_at)