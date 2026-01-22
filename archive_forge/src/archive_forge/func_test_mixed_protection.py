import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_mixed_protection(self):
    self.secret_cls.authorized = True
    response = self.app.get('/secret/1/deepsecret/notfound/', expect_errors=True)
    assert response.status_int == 404
    assert 'secretcontroller' in self.permissions_checked
    assert 'deepsecret' not in self.permissions_checked