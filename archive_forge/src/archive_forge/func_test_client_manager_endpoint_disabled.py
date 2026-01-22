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
def test_client_manager_endpoint_disabled(self):
    auth_args = copy.deepcopy(self.default_password_auth)
    auth_args.update({'user_domain_name': 'default', 'project_domain_name': 'default'})
    client_manager = self._make_clientmanager(auth_args=auth_args, identity_api_version='3', auth_plugin_name='v3password')
    self.assertFalse(client_manager.is_service_available('network'))