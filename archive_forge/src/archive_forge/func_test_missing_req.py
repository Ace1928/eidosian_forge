from unittest import mock
import uuid
import fixtures
import webob
from keystonemiddleware.tests.unit.audit import base
def test_missing_req(self):
    req = webob.Request.blank('http://admin_host:8774/v2/' + str(uuid.uuid4()) + '/servers', environ=self.get_environ_header('GET'))
    req.environ['audit.context'] = {}
    self.assertNotIn('cadf_event', req.environ)
    self.create_simple_middleware()._process_response(req, webob.Response())
    self.assertIn('cadf_event', req.environ)
    payload = req.environ['cadf_event'].as_dict()
    self.assertEqual(payload['outcome'], 'success')
    self.assertEqual(payload['reason']['reasonType'], 'HTTP')
    self.assertEqual(payload['reason']['reasonCode'], '200')
    self.assertEqual(payload['observer']['id'], 'target')