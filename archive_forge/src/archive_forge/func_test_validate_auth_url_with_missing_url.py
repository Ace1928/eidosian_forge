from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
def test_validate_auth_url_with_missing_url(self):
    self.assertRaises(exc.HTTPBadRequest, self.middleware._validate_auth_url, auth_url='')
    self.assertRaises(exc.HTTPBadRequest, self.middleware._validate_auth_url, auth_url=None)