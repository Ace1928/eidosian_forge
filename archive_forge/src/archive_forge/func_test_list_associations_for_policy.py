import uuid
from keystone import exception
def test_list_associations_for_policy(self):
    policy_id = uuid.uuid4().hex
    first = self.create_association(endpoint_id=uuid.uuid4().hex, policy_id=policy_id)
    second = self.create_association(service_id=uuid.uuid4().hex, policy_id=policy_id)
    associations = self.driver.list_associations_for_policy(policy_id)
    self.assertCountEqual([first, second], associations)