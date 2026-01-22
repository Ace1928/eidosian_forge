import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_get_unknown_endpoint(self):
    url = 'http://unknown:8774/v2/' + str(uuid.uuid4()) + '/servers'
    payload = self.get_payload('GET', url)
    self.assertEqual(payload['action'], 'read/list')
    self.assertEqual(payload['outcome'], 'pending')
    self.assertEqual(payload['target']['name'], 'unknown')
    self.assertEqual(payload['target']['id'], 'unknown')
    self.assertEqual(payload['target']['typeURI'], 'unknown')