import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_service_catalog_without_service_type(self):
    token = fixture.V2Token()
    token.set_scope()
    public_urls = []
    for i in range(0, 3):
        public_url = uuid.uuid4().hex
        public_urls.append(public_url)
        s = token.add_service(uuid.uuid4().hex)
        s.add_endpoint(public=public_url)
    auth_ref = access.create(body=token)
    urls = auth_ref.service_catalog.get_urls(service_type=None, interface='public')
    self.assertEqual(3, len(urls))
    for p in public_urls:
        self.assertIn(p, urls)