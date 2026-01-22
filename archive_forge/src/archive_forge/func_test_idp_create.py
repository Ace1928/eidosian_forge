import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
def test_idp_create(self):
    idp_id = uuid.uuid4().hex
    idp = self.client.federation.identity_providers.create(id=idp_id)
    self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
    self.assertEqual(idp_id, idp.id)
    self.assertEqual([], idp.remote_ids)
    self.assertFalse(idp.enabled)