from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_update_account_password_policy(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.update_account_password_policy(minimum_password_length=88)
    self.assert_request_parameters({'Action': 'UpdateAccountPasswordPolicy', 'MinimumPasswordLength': 88}, ignore_params_values=['Version'])