from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_enable_classic_link(self):
    self.set_http_response(status_code=200)
    response = self.vpc.disable_classic_link(dry_run=True)
    self.assertTrue(response)
    self.assert_request_parameters({'Action': 'DisableVpcClassicLink', 'VpcId': self.vpc_id, 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])