import uuid
from keystone import exception
def test_delete_association_by_endpoint(self):
    endpoint_id = uuid.uuid4().hex
    associations = [self.create_association(endpoint_id=endpoint_id), self.create_association(endpoint_id=endpoint_id)]
    self.driver.delete_association_by_endpoint(endpoint_id)
    for association in associations:
        self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)