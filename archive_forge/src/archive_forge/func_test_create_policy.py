from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_create_policy(self):
    self.set_http_response(status_code=200)
    policy_doc = '\n{\n    "Version": "2012-10-17",\n    "Statement": [\n        {\n            "Sid": "Stmt1430948004000",\n            "Effect": "Deny",\n            "Action": [\n                "s3:*"\n            ],\n            "Resource": [\n                "*"\n            ]\n        }\n    ]\n}\n        '
    response = self.service_connection.create_policy('S3-read-only-example-bucket', policy_doc)
    self.assert_request_parameters({'Action': 'CreatePolicy', 'PolicyDocument': policy_doc, 'Path': '/', 'PolicyName': 'S3-read-only-example-bucket'}, ignore_params_values=['Version'])
    self.assertEqual(response['create_policy_response']['create_policy_result']['policy']['policy_name'], 'S3-read-only-example-bucket')