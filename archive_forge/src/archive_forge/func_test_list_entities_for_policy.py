from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_list_entities_for_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.list_entities_for_policy('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket')
    self.assert_request_parameters({'Action': 'ListEntitiesForPolicy', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
    self.assertEqual(len(response['list_entities_for_policy_response']['list_entities_for_policy_result']['policy_roles']), 1)
    self.assertEqual(len(response['list_entities_for_policy_response']['list_entities_for_policy_result']['policy_groups']), 1)
    self.assertEqual(len(response['list_entities_for_policy_response']['list_entities_for_policy_result']['policy_users']), 2)
    self.assertEqual({'user_name': 'Alice'} in response['list_entities_for_policy_response']['list_entities_for_policy_result']['policy_users'], True)