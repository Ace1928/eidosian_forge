import uuid
from keystone import exception
def test_delete_association_by_service(self):
    service_id = uuid.uuid4().hex
    associations = [self.create_association(service_id=service_id), self.create_association(service_id=service_id)]
    self.driver.delete_association_by_service(service_id)
    for association in associations:
        self.assertRaises(exception.PolicyAssociationNotFound, self.driver.check_policy_association, **association)