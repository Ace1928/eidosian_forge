import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def test_invalid_auth_url_v1(self):
    """
        Test that a 400 during authenticate raises exception.AuthBadRequest
        """

    def fake_do_request(*args, **kwargs):
        resp = webob.Response()
        resp.status = http.BAD_REQUEST
        return (FakeResponse(resp), '')
    self.mock_object(auth.KeystoneStrategy, '_do_request', fake_do_request)
    bad_creds = {'username': 'user1', 'auth_url': 'http://localhost/badauthurl/', 'password': 'pass', 'strategy': 'keystone', 'region': 'RegionOne'}
    plugin = auth.KeystoneStrategy(bad_creds)
    self.assertRaises(exception.AuthBadRequest, plugin.authenticate)