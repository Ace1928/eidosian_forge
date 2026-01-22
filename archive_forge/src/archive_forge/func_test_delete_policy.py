from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_delete_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.delete_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
    self.assert_request_parameters({'Action': 'DeletePolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
    self.assertEqual('request_id' in response['delete_policy_response']['response_metadata'], True)