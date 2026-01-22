import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
@classmethod
def independent_check_permissions(cls):
    permissions_checked.add('independent')
    return cls.independent_authorization