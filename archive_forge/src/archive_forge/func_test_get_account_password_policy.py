from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_get_account_password_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_account_password_policy()
    self.assert_request_parameters({'Action': 'GetAccountPasswordPolicy'}, ignore_params_values=['Version'])
    self.assertEquals(response['get_account_password_policy_response']['get_account_password_policy_result']['password_policy']['minimum_password_length'], '12')