import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_post_empty_body_action(self):
    url = 'http://admin_host:8774/v2/%s/servers/action' % uuid.uuid4().hex
    payload = self.get_payload('POST', url)
    self.assertEqual(payload['target']['typeURI'], 'service/compute/servers/action')
    self.assertEqual(payload['action'], 'create')
    self.assertEqual(payload['outcome'], 'pending')