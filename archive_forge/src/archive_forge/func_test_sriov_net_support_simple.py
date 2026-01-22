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
def test_sriov_net_support_simple(self):
    self.set_http_response(status_code=200)
    self.ec2.register_image('name', 'description', image_location='s3://foo', sriov_net_support='simple')
    self.assert_request_parameters({'Action': 'RegisterImage', 'ImageLocation': 's3://foo', 'Name': 'name', 'Description': 'description', 'SriovNetSupport': 'simple'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])