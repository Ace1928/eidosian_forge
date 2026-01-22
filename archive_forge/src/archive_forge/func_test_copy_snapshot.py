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
def test_copy_snapshot(self):
    self.set_http_response(status_code=200)
    snapshot_id = self.ec2.copy_snapshot('us-west-2', 'snap-id', 'description')
    self.assertEqual(snapshot_id, 'snap-copied-id')
    self.assert_request_parameters({'Action': 'CopySnapshot', 'Description': 'description', 'SourceRegion': 'us-west-2', 'SourceSnapshotId': 'snap-id'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])