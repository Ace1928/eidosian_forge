from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_container_acl_remove(self, entity_ref=None):
    entity_ref = entity_ref or self.container_ref
    self.responses.delete(self.container_acl_ref)
    entity = self.manager.create(entity_ref=entity_ref)
    entity.remove()
    self.assertEqual(self.container_acl_ref, self.responses.last_request.url)