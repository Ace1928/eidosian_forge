import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_unlocked_attribute_in_insecure(self):
    response = self.app.get('/notsecret/unlocked/')
    assert response.status_int == 200
    assert response.body == b'Index unlocked'