import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import fixture
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_client_params(self):
    with self.deprecations.expect_deprecations_here():
        sess = session.Session()
        auth = token_endpoint.Token('a', 'b')
    opts = {'auth': auth, 'connect_retries': 50, 'endpoint_override': uuid.uuid4().hex, 'interface': uuid.uuid4().hex, 'region_name': uuid.uuid4().hex, 'service_name': uuid.uuid4().hex, 'user_agent': uuid.uuid4().hex}
    cl = client.Client(session=sess, **opts)
    for k, v in opts.items():
        self.assertEqual(v, getattr(cl._adapter, k))
    self.assertEqual('identity', cl._adapter.service_type)
    self.assertEqual((2, 0), cl._adapter.version)