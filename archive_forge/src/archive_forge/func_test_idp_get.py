import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
def test_idp_get(self):
    idp_id = uuid.uuid4().hex
    remote_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    idp_create = self.client.federation.identity_providers.create(id=idp_id, enabled=True, remote_ids=remote_ids)
    self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
    idp_get = self.client.federation.identity_providers.get(idp_id)
    self.assertEqual(idp_create.id, idp_get.id)
    self.assertEqual(idp_create.enabled, idp_get.enabled)
    self.assertIn(idp_create.remote_ids[0], idp_get.remote_ids)
    self.assertIn(idp_create.remote_ids[1], idp_get.remote_ids)