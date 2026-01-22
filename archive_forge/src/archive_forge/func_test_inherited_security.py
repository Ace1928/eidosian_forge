import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_inherited_security(self):
    assert self.app.get('/secured/', status=401).status_int == 401
    assert self.app.get('/unsecured/').status_int == 200