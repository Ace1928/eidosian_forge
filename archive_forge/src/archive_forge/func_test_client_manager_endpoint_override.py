import copy
from unittest import mock
from keystoneauth1.access import service_catalog
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1.identity import generic as generic_plugin
from keystoneauth1.identity.v3 import k2k
from keystoneauth1 import loading
from keystoneauth1 import noauth
from keystoneauth1 import token_endpoint
from openstack.config import cloud_config
from openstack.config import defaults
from openstack import connection
from osc_lib.api import auth
from osc_lib import clientmanager
from osc_lib import exceptions as exc
from osc_lib.tests import fakes
from osc_lib.tests import utils
def test_client_manager_endpoint_override(self):
    client_manager = self._make_clientmanager(auth_args={}, config_args={'compute_endpoint_override': 'http://example.com', 'foo_bar_endpoint_override': 'http://example2.com'}, auth_plugin_name='none')
    self.assertEqual('http://example.com', client_manager.get_endpoint_for_service_type('compute'))
    self.assertEqual('http://example2.com', client_manager.get_endpoint_for_service_type('foo-bar'))
    self.assertTrue(client_manager.is_service_available('compute'))