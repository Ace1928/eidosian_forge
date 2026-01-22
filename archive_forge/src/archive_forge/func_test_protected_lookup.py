import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_protected_lookup(self):
    response = self.app.get('/secret/hi/', expect_errors=True)
    assert response.status_int == 401
    self.secret_cls.authorized = True
    response = self.app.get('/secret/hi/')
    assert response.status_int == 200
    assert response.body == b'Index hi'
    assert 'secretcontroller' in self.permissions_checked