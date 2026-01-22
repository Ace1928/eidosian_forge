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
def test_reserved_instances_listing(self):
    self.set_http_response(status_code=200)
    response = self.ec2.cancel_reserved_instances_listing()
    self.assertEqual(len(response), 1)
    cancellation = response[0]
    self.assertEqual(cancellation.status, 'cancelled')
    self.assertEqual(cancellation.status_message, 'CANCELLED')
    self.assertEqual(len(cancellation.instance_counts), 4)
    first = cancellation.instance_counts[0]
    self.assertEqual(first.state, 'Available')
    self.assertEqual(first.instance_count, 0)
    self.assertEqual(len(cancellation.price_schedules), 5)
    schedule = cancellation.price_schedules[0]
    self.assertEqual(schedule.term, 5)
    self.assertEqual(schedule.price, '166.64')
    self.assertEqual(schedule.currency_code, 'USD')
    self.assertEqual(schedule.active, False)