import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_wrapped_attribute_failure(self):
    self.secret_cls.independent_authorization = False
    response = self.app.get('/secret/wrapped/', expect_errors=True)
    assert response.status_int == 401
    assert len(self.permissions_checked) == 1
    assert 'independent' in self.permissions_checked