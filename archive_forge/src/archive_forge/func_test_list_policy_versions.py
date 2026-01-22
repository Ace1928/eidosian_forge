from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_list_policy_versions(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.list_policy_versions('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', max_items=3)
    self.assert_request_parameters({'Action': 'ListPolicyVersions', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', 'MaxItems': 3}, ignore_params_values=['Version'])
    self.assertEqual(len(response['list_policy_versions_response']['list_policy_versions_result']['versions']), 3)