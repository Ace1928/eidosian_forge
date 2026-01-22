from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.jsonresponse import ListElement
from boto.ses.connection import SESConnection
def test_ses_get_identity_dkim_list(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_identity_dkim_attributes(['test@amazon.com', 'secondtest@amazon.com'])
    response = response['GetIdentityDkimAttributesResponse']
    result = response['GetIdentityDkimAttributesResult']
    first_entry = result['DkimAttributes'][0]
    entry_key = first_entry['key']
    attributes = first_entry['value']
    tokens = attributes['DkimTokens']
    self.assertEqual(entry_key, 'test@amazon.com')
    self.assertEqual(ListElement, type(tokens))
    self.assertEqual(3, len(tokens))
    self.assertEqual('vvjuipp74whm76gqoni7qmwwn4w4qusjiainivf6f', tokens[0])
    self.assertEqual('3frqe7jn4obpuxjpwpolz6ipb3k5nvt2nhjpik2oy', tokens[1])
    self.assertEqual('wrqplteh7oodxnad7hsl4mixg2uavzneazxv5sxi2', tokens[2])
    second_entry = result['DkimAttributes'][1]
    entry_key = second_entry['key']
    attributes = second_entry['value']
    dkim_enabled = attributes['DkimEnabled']
    dkim_verification_status = attributes['DkimVerificationStatus']
    self.assertEqual(entry_key, 'secondtest@amazon.com')
    self.assertEqual(dkim_enabled, 'false')
    self.assertEqual(dkim_verification_status, 'NotStarted')