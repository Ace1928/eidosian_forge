from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations_with_user_id(self):
    user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
    params = {'user_id': user_id}
    ms = self.cs.migrations.list(**params)
    self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-migrations?user_id=%s' % user_id)
    for m in ms:
        self.assertIsInstance(m, migrations.Migration)