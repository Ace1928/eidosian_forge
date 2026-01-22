import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_build_userdata(self):
    """Tests the build_userdata function."""
    cfg.CONF.set_override('heat_metadata_server_url', 'http://server.test:123')
    cfg.CONF.set_override('instance_connection_is_secure', False)
    cfg.CONF.set_override('instance_connection_https_validate_certificates', False)
    data = self.nova_plugin.build_userdata({})
    self.assertIn('Content-Type: text/cloud-config;', data)
    self.assertIn('Content-Type: text/cloud-boothook;', data)
    self.assertIn('Content-Type: text/part-handler;', data)
    self.assertIn('Content-Type: text/x-cfninitdata;', data)
    self.assertIn('Content-Type: text/x-shellscript;', data)
    self.assertIn('http://server.test:123', data)
    self.assertIn('[Boto]', data)