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
def test_copy_image_required_params(self):
    self.set_http_response(status_code=200)
    copied_ami = self.ec2.copy_image('us-west-2', 'ami-id')
    self.assertEqual(copied_ami.image_id, 'ami-copied-id')
    self.assert_request_parameters({'Action': 'CopyImage', 'SourceRegion': 'us-west-2', 'SourceImageId': 'ami-id'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])