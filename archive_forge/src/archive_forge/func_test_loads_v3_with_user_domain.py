import uuid
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1.loading._plugins.identity import generic
from keystoneauth1 import session
from keystoneauth1.tests.unit.loading import utils
def test_loads_v3_with_user_domain(self):
    auth_url = 'http://keystone.test:5000'
    disc = fixture.DiscoveryList(href=auth_url)
    sess = session.Session()
    self.requests_mock.get(auth_url, json=disc)
    plugin = generic.Password().load_from_options(auth_url=auth_url, user_id=uuid.uuid4().hex, password=uuid.uuid4().hex, project_id=uuid.uuid4().hex, user_domain_id=uuid.uuid4().hex)
    inner_plugin = plugin._do_create_plugin(sess)
    self.assertIsInstance(inner_plugin, identity.V3Password)
    self.assertEqual(inner_plugin.auth_url, auth_url + '/v3')