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
def test_get_all_snapshots(self):
    self.set_http_response(status_code=200)
    result = self.ec2.get_all_snapshots(snapshot_ids=['snap-1a2b3c4d', 'snap-5e6f7a8b'])
    self.assert_request_parameters({'Action': 'DescribeSnapshots', 'SnapshotId.1': 'snap-1a2b3c4d', 'SnapshotId.2': 'snap-5e6f7a8b'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0].id, 'snap-1a2b3c4d')
    self.assertTrue(result[0].encrypted)
    self.assertEqual(result[1].id, 'snap-5e6f7a8b')
    self.assertFalse(result[1].encrypted)