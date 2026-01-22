import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_unauthorized_client_requests(self):
    with self.deprecations.expect_deprecations_here():
        cl = get_client()
    self.assertRaises(exceptions.AuthorizationFailure, cl.get, '/hi')
    self.assertRaises(exceptions.AuthorizationFailure, cl.post, '/hi')
    self.assertRaises(exceptions.AuthorizationFailure, cl.put, '/hi')
    self.assertRaises(exceptions.AuthorizationFailure, cl.delete, '/hi')