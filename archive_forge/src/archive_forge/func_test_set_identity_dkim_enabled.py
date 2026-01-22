from tests.unit import unittest
from boto.ses.connection import SESConnection
from boto.ses import exceptions
def test_set_identity_dkim_enabled(self):
    with self.assertRaises(exceptions.SESIdentityNotVerifiedError):
        self.ses.set_identity_dkim_enabled('example.com', True)