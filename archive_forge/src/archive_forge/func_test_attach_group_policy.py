from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_attach_group_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.attach_group_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'Dev')
    self.assert_request_parameters({'Action': 'AttachGroupPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'GroupName': 'Dev'}, ignore_params_values=['Version'])
    self.assertEqual('request_id' in response['attach_group_policy_response']['response_metadata'], True)