from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations_with_changes_before(self):
    params = {'changes_before': '2012-02-29T06:23:22'}
    ms = self.cs.migrations.list(**params)
    self.assert_request_id(ms, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-migrations?changes-before=%s' % '2012-02-29T06%3A23%3A22')
    for m in ms:
        self.assertIsInstance(m, migrations.Migration)