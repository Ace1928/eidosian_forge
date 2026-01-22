import uuid
from pycadf import cadftaxonomy as taxonomy
import webob
from keystonemiddleware import audit
from keystonemiddleware.tests.unit.audit import base
def test_request_and_global_request_id(self):
    path = '/v1/' + str(uuid.uuid4())
    url = 'https://23.253.72.207' + path
    request_id = 'req-%s' % uuid.uuid4()
    global_request_id = 'req-%s' % uuid.uuid4()
    env_headers = self.get_environ_header('GET')
    env_headers['openstack.request_id'] = request_id
    env_headers['openstack.global_request_id'] = global_request_id
    payload = self.get_payload('GET', url, environ=env_headers)
    self.assertEqual(payload['initiator']['request_id'], request_id)
    self.assertEqual(payload['initiator']['global_request_id'], global_request_id)
    payload = self.get_payload('GET', url)
    self.assertNotIn('request_id', payload['initiator'])
    self.assertNotIn('global_request_id', payload['initiator'])