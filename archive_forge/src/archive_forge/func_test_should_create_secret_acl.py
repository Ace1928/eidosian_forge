from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def test_should_create_secret_acl(self):
    entity = self.manager.create(entity_ref=self.secret_ref + '///', users=self.users1, project_access=True)
    self.assertIsInstance(entity, acls.SecretACL)
    read_acl = entity.read
    self.assertEqual(self.secret_ref + '///', read_acl.entity_ref)
    self.assertTrue(read_acl.project_access)
    self.assertEqual(self.users1, read_acl.users)
    self.assertEqual(acls.DEFAULT_OPERATION_TYPE, read_acl.operation_type)
    self.assertIn(self.secret_ref, read_acl.acl_ref, 'ACL ref has additional /acl')
    self.assertIsNone(read_acl.created)
    self.assertIsNone(read_acl.updated)
    read_acl_via_get = entity.get('read')
    self.assertEqual(read_acl, read_acl_via_get)