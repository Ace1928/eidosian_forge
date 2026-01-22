import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_unlocked_attribute(self):

    class AuthorizedSubController(object):

        @expose()
        def index(self):
            return 'Index'

        @expose()
        def allowed(self):
            return 'Allowed!'

    class SecretController(SecureController):

        @expose()
        def index(self):
            return 'Index'

        @expose()
        @unlocked
        def allowed(self):
            return 'Allowed!'
        authorized = unlocked(AuthorizedSubController())

    class RootController(object):

        @expose()
        def index(self):
            return 'Hello, World!'

        @expose()
        @secure(lambda: False)
        def locked(self):
            return 'No dice!'

        @expose()
        @secure(lambda: True)
        def unlocked(self):
            return 'Sure thing'
        secret = SecretController()
    app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    response = app.get('/unlocked')
    assert response.status_int == 200
    assert response.body == b'Sure thing'
    response = app.get('/locked', expect_errors=True)
    assert response.status_int == 401
    response = app.get('/secret/', expect_errors=True)
    assert response.status_int == 401
    response = app.get('/secret/allowed')
    assert response.status_int == 200
    assert response.body == b'Allowed!'
    response = app.get('/secret/authorized/')
    assert response.status_int == 200
    assert response.body == b'Index'
    response = app.get('/secret/authorized/allowed')
    assert response.status_int == 200
    assert response.body == b'Allowed!'