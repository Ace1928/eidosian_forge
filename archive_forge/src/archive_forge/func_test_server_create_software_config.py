from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from urllib import parse as urlparse
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import environment
from heat.engine.resources.openstack.heat import deployed_server
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
@mock.patch.object(heat_plugin.HeatClientPlugin, 'url_for')
def test_server_create_software_config(self, fake_url):
    fake_url.return_value = 'the-cfn-url'
    server = self._server_create_software_config()
    self.assertEqual({'os-collect-config': {'cfn': {'access_key_id': '4567', 'metadata_url': 'the-cfn-url/v1/', 'path': 'server.Metadata', 'secret_access_key': '8901', 'stack_name': 'server_sc_s'}, 'collectors': ['cfn', 'local']}, 'deployments': []}, server.metadata_get())