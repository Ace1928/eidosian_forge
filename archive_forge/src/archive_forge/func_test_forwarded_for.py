import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_forwarded_for(self):
    ORIGINAL_IP = '10.100.100.1'
    with self.deprecations.expect_deprecations_here():
        cl = httpclient.HTTPClient(username='username', password='password', project_id='tenant', auth_url='auth_test', original_ip=ORIGINAL_IP)
    self.stub_url('GET')
    with self.deprecations.expect_deprecations_here():
        cl.request(self.TEST_URL, 'GET')
    forwarded = 'for=%s;by=%s' % (ORIGINAL_IP, httpclient.USER_AGENT)
    self.assertRequestHeaderEqual('Forwarded', forwarded)