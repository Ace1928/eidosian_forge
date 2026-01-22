import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_independent_check_success(self):
    self.secret_cls.independent_authorization = True
    response = self.app.get('/secret/independent')
    assert response.status_int == 200
    assert response.body == b'Independent Security'
    assert len(self.permissions_checked) == 1
    assert 'independent' in self.permissions_checked