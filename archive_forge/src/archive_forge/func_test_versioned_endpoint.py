import testtools
from glanceclient import client
from glanceclient import v1
from glanceclient import v2
def test_versioned_endpoint(self):
    gc = client.Client(1, 'http://example.com/v2')
    self.assertEqual('http://example.com', gc.http_client.endpoint)
    self.assertIsInstance(gc, v1.client.Client)