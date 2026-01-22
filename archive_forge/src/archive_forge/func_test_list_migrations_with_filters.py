from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations_with_filters(self):
    ml = self.cs.migrations.list('host1', 'finished')
    self.assert_request_id(ml, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-migrations?host=host1&status=finished')
    for m in ml:
        self.assertIsInstance(m, migrations.Migration)