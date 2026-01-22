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
def test_serialized_api_args(self):
    self.set_http_response(status_code=200)
    response = self.ec2.describe_reserved_instances_modifications(reserved_instances_modification_ids=['2567o137-8a55-48d6-82fb-7258506bb497'], filters={'status': 'processing'})
    self.assert_request_parameters({'Action': 'DescribeReservedInstancesModifications', 'Filter.1.Name': 'status', 'Filter.1.Value.1': 'processing', 'ReservedInstancesModificationId.1': '2567o137-8a55-48d6-82fb-7258506bb497'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(response[0].modification_id, 'rimod-49b9433e-fdc7-464a-a6e5-9dabcexample')
    self.assertEqual(response[0].create_date, datetime(2013, 9, 2, 21, 20, 19, 637000))
    self.assertEqual(response[0].update_date, datetime(2013, 9, 2, 21, 38, 24, 143000))
    self.assertEqual(response[0].effective_date, datetime(2013, 9, 2, 21, 0, 0, 0))
    self.assertEqual(response[0].status, 'fulfilled')
    self.assertEqual(response[0].status_message, None)
    self.assertEqual(response[0].client_token, 'token-f5b56c05-09b0-4d17-8d8c-c75d8a67b806')
    self.assertEqual(response[0].reserved_instances[0].id, '2567o137-8a55-48d6-82fb-7258506bb497')
    self.assertEqual(response[0].modification_results[0].availability_zone, 'us-east-1b')
    self.assertEqual(response[0].modification_results[0].platform, 'EC2-VPC')
    self.assertEqual(response[0].modification_results[0].instance_count, 1)
    self.assertEqual(len(response), 1)