from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations_with_instance_uuid_filter(self):
    ml = self.cs.migrations.list('host1', 'finished', 'instance_id_456')
    self.assert_request_id(ml, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-migrations?host=host1&instance_uuid=instance_id_456&status=finished')
    self.assertEqual(1, len(ml))
    self.assertEqual('instance_id_456', ml[0].instance_uuid)