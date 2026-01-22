import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
def test_secure_obj_only_failure(self):

    class Foo(object):
        pass
    try:
        secure(Foo())
    except Exception as e:
        assert isinstance(e, TypeError)