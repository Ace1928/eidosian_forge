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
def test_get_reserved_instance_offerings_params(self):
    self.set_http_response(status_code=200)
    self.ec2.get_all_reserved_instances_offerings(reserved_instances_offering_ids=['id1', 'id2'], instance_type='t1.micro', availability_zone='us-east-1', product_description='description', instance_tenancy='dedicated', offering_type='offering_type', include_marketplace=False, min_duration=100, max_duration=1000, max_instance_count=1, next_token='next_token', max_results=10)
    self.assert_request_parameters({'Action': 'DescribeReservedInstancesOfferings', 'ReservedInstancesOfferingId.1': 'id1', 'ReservedInstancesOfferingId.2': 'id2', 'InstanceType': 't1.micro', 'AvailabilityZone': 'us-east-1', 'ProductDescription': 'description', 'InstanceTenancy': 'dedicated', 'OfferingType': 'offering_type', 'IncludeMarketplace': 'false', 'MinDuration': '100', 'MaxDuration': '1000', 'MaxInstanceCount': '1', 'NextToken': 'next_token', 'MaxResults': '10'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])