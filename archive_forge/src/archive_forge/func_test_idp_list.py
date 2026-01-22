import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
def test_idp_list(self):
    idp_ids = []
    for _ in range(3):
        idp_id = uuid.uuid4().hex
        self.client.federation.identity_providers.create(id=idp_id)
        self.addCleanup(self.client.federation.identity_providers.delete, idp_id)
        idp_ids.append(idp_id)
    idp_list = self.client.federation.identity_providers.list()
    fetched_ids = [fetched_idp.id for fetched_idp in idp_list]
    for idp_id in idp_ids:
        self.assertIn(idp_id, fetched_ids)