from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_set_default_policy_version(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.set_default_policy_version('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'v1')
    self.assert_request_parameters({'Action': 'SetDefaultPolicyVersion', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'VersionId': 'v1'}, ignore_params_values=['Version'])
    self.assertEqual('request_id' in response['set_default_policy_version_response']['response_metadata'], True)