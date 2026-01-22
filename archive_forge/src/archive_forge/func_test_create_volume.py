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
def test_create_volume(self):
    self.set_http_response(status_code=200)
    result = self.ec2.create_volume(80, 'us-east-1e', snapshot='snap-1a2b3c4d', encrypted=True)
    self.assert_request_parameters({'Action': 'CreateVolume', 'AvailabilityZone': 'us-east-1e', 'Size': 80, 'SnapshotId': 'snap-1a2b3c4d', 'Encrypted': 'true'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(result.id, 'vol-1a2b3c4d')
    self.assertTrue(result.encrypted)