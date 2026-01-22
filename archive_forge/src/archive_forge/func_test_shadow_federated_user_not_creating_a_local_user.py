import copy
import uuid
from keystone.common import driver_hints
from keystone.common import provider_api
def test_shadow_federated_user_not_creating_a_local_user(self):
    federated_user1 = copy.deepcopy(self.federated_user)
    ShadowUsersCoreTests.normalize_federated_user_properties_for_test(federated_user1, email='some_id@mail.provider')
    PROVIDERS.identity_api.shadow_federated_user(federated_user1['idp_id'], federated_user1['protocol_id'], federated_user1)
    hints = driver_hints.Hints()
    hints.add_filter('name', federated_user1['display_name'])
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))
    federated_user2 = copy.deepcopy(federated_user1)
    federated_user2['name'] = uuid.uuid4().hex
    federated_user2['id'] = uuid.uuid4().hex
    federated_user2['email'] = 'some_id_2@mail.provider'
    PROVIDERS.identity_api.shadow_federated_user(federated_user2['idp_id'], federated_user2['protocol_id'], federated_user2)
    hints.add_filter('name', federated_user2['display_name'])
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(1, len(users))