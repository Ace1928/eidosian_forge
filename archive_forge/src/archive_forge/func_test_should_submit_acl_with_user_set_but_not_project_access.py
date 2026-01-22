from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_submit_acl_with_user_set_but_not_project_access(self):
    data = {'acl_ref': self.container_acl_ref}
    self.responses.put(self.container_acl_ref, json=data)
    entity = self.manager.create(entity_ref=self.container_ref, users=self.users2)
    api_resp = entity.submit()
    self.assertEqual(self.container_acl_ref, api_resp)
    self.assertEqual(self.container_acl_ref, self.responses.last_request.url)
    self.assertEqual(self.users2, entity.read.users)
    self.assertIsNone(entity.get('read').project_access)