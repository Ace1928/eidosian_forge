import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_secure_attribute(self):
    authorized = False

    class SubController(object):

        @expose()
        def index(self):
            return 'Hello from sub!'

    class RootController(object):

        @expose()
        def index(self):
            return 'Hello from root!'
        sub = secure(SubController(), lambda: authorized)
    app = TestApp(make_app(RootController()))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello from root!'
    response = app.get('/sub/', expect_errors=True)
    assert response.status_int == 401
    authorized = True
    response = app.get('/sub/')
    assert response.status_int == 200
    assert response.body == b'Hello from sub!'