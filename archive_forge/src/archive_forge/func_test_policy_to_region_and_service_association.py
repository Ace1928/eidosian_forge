import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def test_policy_to_region_and_service_association(self):
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[0]['id'], service_id=self.service[0]['id'], region_id=self.region[0]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[1]['id'], service_id=self.service[1]['id'], region_id=self.region[1]['id'])
    PROVIDERS.endpoint_policy_api.create_policy_association(self.policy[2]['id'], service_id=self.service[2]['id'], region_id=self.region[2]['id'])
    self._assert_correct_policy(self.endpoint[0], self.policy[0])
    self._assert_correct_policy(self.endpoint[5], self.policy[0])
    self._assert_correct_endpoints(self.policy[2], [self.endpoint[4]])
    self._assert_correct_endpoints(self.policy[1], [self.endpoint[2]])
    self._assert_correct_endpoints(self.policy[0], [self.endpoint[0], self.endpoint[5]])