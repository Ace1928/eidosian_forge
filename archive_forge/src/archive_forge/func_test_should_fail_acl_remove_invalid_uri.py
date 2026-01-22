from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_fail_acl_remove_invalid_uri(self):
    entity = self.manager.create(entity_ref=self.secret_acl_ref)
    self.assertRaises(ValueError, entity.remove)
    entity = self.manager.create(entity_ref=self.container_acl_ref)
    self.assertRaises(ValueError, entity.remove)
    entity = self.manager.create(entity_ref=self.container_ref + '/consumers')
    self.assertRaises(ValueError, entity.remove)
    entity = self.manager.create(entity_ref=self.endpoint + '/secrets' + '/consumers')
    self.assertRaises(ValueError, entity.remove)