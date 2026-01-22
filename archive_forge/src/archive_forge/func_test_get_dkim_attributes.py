from tests.unit import unittest
from boto.ses.connection import SESConnection
from boto.ses import exceptions
def test_get_dkim_attributes(self):
    response = self.ses.get_identity_dkim_attributes(['example.com'])
    self.assertTrue('GetIdentityDkimAttributesResponse' in response)
    self.assertTrue('GetIdentityDkimAttributesResult' in response['GetIdentityDkimAttributesResponse'])
    self.assertTrue('DkimAttributes' in response['GetIdentityDkimAttributesResponse']['GetIdentityDkimAttributesResult'])