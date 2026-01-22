from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
def test_get_all_classic_link_vpcs_params(self):
    self.set_http_response(status_code=200)
    self.service_connection.get_all_classic_link_vpcs(vpc_ids=['id1', 'id2'], filters={'GroupId': 'sg-9b4343fe'}, dry_run=True)
    self.assert_request_parameters({'Action': 'DescribeVpcClassicLink', 'VpcId.1': 'id1', 'VpcId.2': 'id2', 'Filter.1.Name': 'GroupId', 'Filter.1.Value.1': 'sg-9b4343fe', 'DryRun': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])