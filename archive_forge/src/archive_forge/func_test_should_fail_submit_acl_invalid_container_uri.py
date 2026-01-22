from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_submit_acl_invalid_container_uri(self):
    """Adding tests for container URI validation.

        Container URI validation is different from secret URI validation.
        That's why adding separate tests for code coverage.
        """
    data = {'acl_ref': self.container_acl_ref}
    self.responses.put(self.container_acl_ref, json=data)
    entity = self.manager.create(entity_ref=self.container_acl_ref + '///', users=self.users1, project_access=True)
    self.assertRaises(ValueError, entity.submit)
    entity = self.manager.create(entity_ref=self.container_ref, users=self.users1, project_access=True)
    entity._entity_ref = None
    self.assertRaises(ValueError, entity.submit)
    entity = self.manager.create(entity_ref=self.container_ref, users=self.users1, project_access=True)
    entity._entity_ref = self.secret_ref
    self.assertRaises(ValueError, entity.submit)