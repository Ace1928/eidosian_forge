import uuid
from keystone import exception
def test_get_policy_association(self):
    association = self.create_association(service_id=uuid.uuid4().hex)
    policy_id = association.pop('policy_id')
    association_ref = self.driver.get_policy_association(**association)
    self.assertEqual({'policy_id': (policy_id,)}, association_ref)