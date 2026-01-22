from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_list_policies(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.list_policies(max_items=4)
    self.assert_request_parameters({'Action': 'ListPolicies', 'MaxItems': 4}, ignore_params_values=['Version'])
    self.assertEqual(len(response['list_policies_response']['list_policies_result']['policies']), 4)
    self.assertEqual(response['list_policies_response']['list_policies_result']['is_truncated'], 'true')