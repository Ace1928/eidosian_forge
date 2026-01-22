from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_submit_acl_with_project_access_set_but_no_users(self):
    data = {'acl_ref': self.secret_acl_ref}
    self.responses.put(self.secret_acl_ref, json=data)
    entity = self.manager.create(entity_ref=self.secret_ref, project_access=False)
    api_resp = entity.submit()
    self.assertEqual(self.secret_acl_ref, api_resp)
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.assertFalse(entity.read.project_access)
    self.assertEqual([], entity.get('read').users)