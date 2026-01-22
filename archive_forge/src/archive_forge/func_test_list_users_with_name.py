import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_list_users_with_name(self):
    self._build_fed_resource()
    federated_dict_1 = unit.new_federated_user_ref(display_name='test1@federation.org')
    federated_dict_2 = unit.new_federated_user_ref(display_name='test2@federation.org')
    domain = self._get_domain_fixture()
    hints = driver_hints.Hints()
    hints.add_filter('name', 'test1@federation.org')
    users = self.identity_api.list_users(hints=hints)
    self.assertEqual(0, len(users))
    self.shadow_users_api.create_federated_user(domain['id'], federated_dict_1)
    self.shadow_users_api.create_federated_user(domain['id'], federated_dict_2)
    hints = driver_hints.Hints()
    hints.add_filter('name', 'test1@federation.org')
    users = self.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))
    hints = driver_hints.Hints()
    hints.add_filter('name', 'test1@federation.org')
    hints.add_filter('idp_id', 'ORG_IDP')
    users = self.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))