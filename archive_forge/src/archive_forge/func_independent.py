import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
@secure('independent_check_permissions')
@expose()
def independent(self):
    return 'Independent Security'