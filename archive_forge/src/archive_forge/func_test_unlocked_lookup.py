import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_unlocked_lookup(self):
    response = self.app.get('/notsecret/1/deepsecret/2/')
    assert response.status_int == 200
    assert response.body == b'Index 2'
    assert 'deepsecret' not in self.permissions_checked
    response = self.app.get('/notsecret/1/deepsecret/notfound/', expect_errors=True)
    assert response.status_int == 404
    assert 'deepsecret' not in self.permissions_checked