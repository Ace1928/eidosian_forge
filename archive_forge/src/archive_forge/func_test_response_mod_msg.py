import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_response_mod_msg(self):
    url = 'http://admin_host:8774/v2/' + str(uuid.uuid4()) + '/servers'
    req = webob.Request.blank(url, environ=self.get_environ_header('GET'), remote_addr='192.168.0.1')
    req.environ['audit.context'] = {}
    middleware = self.create_simple_middleware()
    middleware._process_request(req)
    payload = req.environ['cadf_event'].as_dict()
    middleware._process_response(req, webob.Response())
    payload2 = req.environ['cadf_event'].as_dict()
    self.assertEqual(payload['id'], payload2['id'])
    self.assertEqual(payload['tags'], payload2['tags'])
    self.assertEqual(payload2['outcome'], 'success')
    self.assertEqual(payload2['reason']['reasonType'], 'HTTP')
    self.assertEqual(payload2['reason']['reasonCode'], '200')
    self.assertEqual(len(payload2['reporterchain']), 1)
    self.assertEqual(payload2['reporterchain'][0]['role'], 'modifier')
    self.assertEqual(payload2['reporterchain'][0]['reporter']['id'], 'target')