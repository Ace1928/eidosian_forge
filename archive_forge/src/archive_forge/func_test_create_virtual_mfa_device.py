from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_create_virtual_mfa_device(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.create_virtual_mfa_device('/', 'ExampleName')
    self.assert_request_parameters({'Path': '/', 'VirtualMFADeviceName': 'ExampleName', 'Action': 'CreateVirtualMFADevice'}, ignore_params_values=['Version'])
    self.assertEquals(response['create_virtual_mfa_device_response']['create_virtual_mfa_device_result']['virtual_mfa_device']['serial_number'], 'arn:aws:iam::123456789012:mfa/ExampleName')