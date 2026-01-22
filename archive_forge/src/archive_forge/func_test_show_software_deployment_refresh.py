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
@mock.patch.object(service_software_config.SoftwareConfigService, '_refresh_swift_software_deployment')
def test_show_software_deployment_refresh(self, _refresh_swift_software_deployment):
    temp_url = 'http://192.0.2.1/v1/AUTH_a/b/c?temp_url_sig=ctemp_url_expires=1234'
    config = self._create_software_config(inputs=[{'name': 'deploy_signal_transport', 'type': 'String', 'value': 'TEMP_URL_SIGNAL'}, {'name': 'deploy_signal_id', 'type': 'String', 'value': temp_url}])
    deployment = self._create_software_deployment(status='IN_PROGRESS', config_id=config['id'])
    deployment_id = deployment['id']
    sd = software_deployment_object.SoftwareDeployment.get_by_id(self.ctx, deployment_id)
    _refresh_swift_software_deployment.return_value = sd
    self.assertEqual(deployment, self.engine.show_software_deployment(self.ctx, deployment_id))
    self.assertEqual((self.ctx, sd, temp_url), _refresh_swift_software_deployment.call_args[0])