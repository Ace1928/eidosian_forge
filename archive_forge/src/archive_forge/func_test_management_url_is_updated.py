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
def test_management_url_is_updated(self):
    first = fixture.V2Token()
    first.set_scope()
    admin_url = 'http://admin:35357/v2.0'
    second_url = 'http://secondurl:35357/v2.0'
    s = first.add_service('identity')
    s.add_endpoint(public='http://public.com:5000/v2.0', admin=admin_url)
    second = fixture.V2Token()
    second.set_scope()
    s = second.add_service('identity')
    s.add_endpoint(public='http://secondurl:5000/v2.0', admin=second_url)
    self.stub_auth(response_list=[{'json': first}, {'json': second}])
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL)
    self.assertEqual(cl.management_url, admin_url)
    with self.deprecations.expect_deprecations_here():
        cl.authenticate()
    self.assertEqual(cl.management_url, second_url)