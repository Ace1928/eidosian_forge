import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_post_action(self):
    url = 'http://admin_host:8774/v2/%s/servers/action' % uuid.uuid4().hex
    body = b'{"createImage" : {"name" : "new-image","metadata": {"ImageType": "Gold","ImageVersion": "2.0"}}}'
    payload = self.get_payload('POST', url, body=body)
    self.assertEqual(payload['target']['typeURI'], 'service/compute/servers/action')
    self.assertEqual(payload['action'], 'update/createImage')
    self.assertEqual(payload['outcome'], 'pending')