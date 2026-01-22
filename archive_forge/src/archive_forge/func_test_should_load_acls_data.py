from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_load_acls_data(self):
    self.responses.get(self.container_acl_ref, json=self.get_acl_response_data(users=self.users2, project_access=True))
    entity = self.manager.create(entity_ref=self.container_ref, users=self.users1)
    self.assertEqual(self.container_ref, entity.entity_ref)
    self.assertEqual(self.container_acl_ref, entity.acl_ref)
    entity.load_acls_data()
    self.assertEqual(self.users2, entity.read.users)
    self.assertTrue(entity.get('read').project_access)
    self.assertEqual(timeutils.parse_isotime(self.created), entity.read.created)
    self.assertEqual(timeutils.parse_isotime(self.created), entity.get('read').created)
    self.assertEqual(1, len(entity.operation_acls))
    self.assertEqual(self.container_acl_ref, entity.get('read').acl_ref)
    self.assertEqual(self.container_ref, entity.read.entity_ref)