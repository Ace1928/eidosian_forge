import copy
import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def test_management_url_is_updated_with_domain(self):
    self._management_url_is_updated(client_fixtures.domain_scoped_token(), domain_name='exampledomain')