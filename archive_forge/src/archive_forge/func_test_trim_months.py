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
def test_trim_months(self):
    """
        Test trimming monthly snapshots and ensure that older months
        get deleted properly. The result of this test should be that
        the two oldest snapshots get deleted.
        """
    orig = {'get_all_snapshots': self.ec2.get_all_snapshots, 'delete_snapshot': self.ec2.delete_snapshot}
    snaps = self._get_snapshots()
    self.ec2.get_all_snapshots = MagicMock(return_value=snaps)
    self.ec2.delete_snapshot = MagicMock()
    self.ec2.trim_snapshots(monthly_backups=1)
    self.assertEqual(True, self.ec2.get_all_snapshots.called)
    self.assertEqual(2, self.ec2.delete_snapshot.call_count)
    self.ec2.get_all_snapshots = orig['get_all_snapshots']
    self.ec2.delete_snapshot = orig['delete_snapshot']