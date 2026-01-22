from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
@mock.patch.object(auth_url.cfg, 'CONF')
def test_validate_auth_url_with_valid_url(self, mock_cfg):
    mock_cfg.auth_password.allowed_auth_uris = ['foobar']
    self.assertTrue(self.middleware._validate_auth_url('foobar'))