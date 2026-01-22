import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_layered_protection(self):
    response = self.app.get('/secret/hi/deepsecret/', expect_errors=True)
    assert response.status_int == 401
    assert 'secretcontroller' in self.permissions_checked
    self.secret_cls.authorized = True
    response = self.app.get('/secret/hi/deepsecret/', expect_errors=True)
    assert response.status_int == 401
    assert 'secretcontroller' in self.permissions_checked
    assert 'deepsecret' in self.permissions_checked
    self.deepsecret_cls.authorized = True
    response = self.app.get('/secret/hi/deepsecret/')
    assert response.status_int == 200
    assert response.body == b'Deep Secret'
    assert 'secretcontroller' in self.permissions_checked
    assert 'deepsecret' in self.permissions_checked