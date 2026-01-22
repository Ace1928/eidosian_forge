from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_create_policy_version(self):
    self.set_http_response(status_code=200)
    policy_doc = '\n{\n    "Version": "2012-10-17",\n    "Statement": [\n        {\n            "Sid": "Stmt1430948004000",\n            "Effect": "Deny",\n            "Action": [\n                "s3:*"\n            ],\n            "Resource": [\n                "*"\n            ]\n        }\n    ]\n}\n        '
    response = self.service_connection.create_policy_version('arn:aws:iam::123456789012:policy/S3-read-only-example-bucket', policy_doc, set_as_default=True)
    self.assert_request_parameters({'Action': 'CreatePolicyVersion', 'PolicyDocument': policy_doc, 'SetAsDefault': 'true', 'PolicyArn': 'arn:aws:iam::123456789012:policy/S3-read-only-example-bucket'}, ignore_params_values=['Version'])
    self.assertEqual(response['create_policy_version_response']['create_policy_version_result']['policy_version']['is_default_version'], 'true')