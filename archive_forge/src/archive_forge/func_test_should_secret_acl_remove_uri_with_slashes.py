from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_secret_acl_remove_uri_with_slashes(self):
    self.responses.delete(self.secret_acl_ref)
    entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users2)
    entity.remove()
    self.assertEqual(self.secret_acl_ref, self.responses.last_request.url)
    self.responses.delete(self.container_acl_ref)