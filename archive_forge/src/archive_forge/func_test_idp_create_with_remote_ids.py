import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
def test_idp_create_with_remote_ids(self):
    idp_id = uuid.uuid4().hex
    remote_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    idp = self.client.federation.identity_providers.create(id=idp_id, enabled=True, remote_ids=remote_ids)
    self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
    self.assertEqual(idp_id, idp.id)
    self.assertIn(remote_ids[0], idp.remote_ids)
    self.assertIn(remote_ids[1], idp.remote_ids)
    self.assertTrue(idp.enabled)