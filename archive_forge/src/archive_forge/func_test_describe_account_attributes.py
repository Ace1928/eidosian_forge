from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
def test_describe_account_attributes(self):
    self.set_http_response(status_code=200)
    parsed = self.ec2.describe_account_attributes()
    self.assertEqual(len(parsed), 4)
    self.assertEqual(parsed[0].attribute_name, 'vpc-max-security-groups-per-interface')
    self.assertEqual(parsed[0].attribute_values, ['5'])
    self.assertEqual(parsed[-1].attribute_name, 'default-vpc')
    self.assertEqual(parsed[-1].attribute_values, ['none'])