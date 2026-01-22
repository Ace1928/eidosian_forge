from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import migrations
def test_list_migrations_with_user_id_pre_v280(self):
    self.cs.api_version = api_versions.APIVersion('2.79')
    user_id = '13cc0930d27c4be0acc14d7c47a3e1f7'
    ex = self.assertRaises(TypeError, self.cs.migrations.list, user_id=user_id)
    self.assertIn("unexpected keyword argument 'user_id'", str(ex))