import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_versioned_endpoint_with_minor_revision(self):
    gc = client.Client(2.2, 'http://example.com/v2.1')
    self.assertEqual('http://example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v2.client.Client)