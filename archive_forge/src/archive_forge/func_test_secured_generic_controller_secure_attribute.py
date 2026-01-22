import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_secured_generic_controller_secure_attribute(self):
    authorized = False

    class SecureController(object):

        @expose(generic=True)
        def index(self):
            return 'I should not be allowed'

        @index.when(method='POST')
        def index_post(self):
            return 'I should not be allowed'

        @expose(generic=True)
        def secret(self):
            return 'I should not be allowed'

    class RootController(object):
        sub = secure(SecureController(), lambda: authorized)
    app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
    response = app.get('/sub/', expect_errors=True)
    assert response.status_int == 401
    response = app.post('/sub/', expect_errors=True)
    assert response.status_int == 401
    response = app.get('/sub/secret/', expect_errors=True)
    assert response.status_int == 401