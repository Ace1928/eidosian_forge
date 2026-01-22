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
def test_client_manager_password_verify_insecure_ca(self):
    config_args = {'insecure': True, 'cacert': 'cafile'}
    client_manager = self._make_clientmanager(config_args=config_args, auth_required=True)
    self.assertFalse(client_manager.verify)
    self.assertIsNone(client_manager.cacert)
    self.assertTrue(client_manager.is_service_available('network'))