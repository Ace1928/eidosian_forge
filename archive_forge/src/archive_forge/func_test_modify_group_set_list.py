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
def test_modify_group_set_list(self):
    self.set_http_response(status_code=200)
    self.ec2.modify_network_interface_attribute('id', 'groupSet', ['sg-1', 'sg-2'])
    self.assert_request_parameters({'Action': 'ModifyNetworkInterfaceAttribute', 'NetworkInterfaceId': 'id', 'SecurityGroupId.1': 'sg-1', 'SecurityGroupId.2': 'sg-2'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])