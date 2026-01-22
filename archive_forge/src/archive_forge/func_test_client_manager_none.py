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
def test_client_manager_none(self):
    none_auth = {'endpoint': fakes.AUTH_URL}
    client_manager = self._make_clientmanager(auth_args=none_auth, auth_plugin_name='none')
    self.assertEqual(fakes.AUTH_URL, client_manager._cli_options.config['auth']['endpoint'])
    self.assertIsInstance(client_manager.auth, noauth.NoAuth)
    self.assertEqual(fakes.AUTH_URL, client_manager.get_endpoint_for_service_type('baremetal'))