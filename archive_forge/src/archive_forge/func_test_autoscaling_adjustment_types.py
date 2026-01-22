import base64
from datetime import datetime
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.autoscale import AutoScaleConnection
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.tag import Tag
from boto.ec2.blockdevicemapping import EBSBlockDeviceType, BlockDeviceMapping
from boto.ec2.autoscale import launchconfig, LaunchConfiguration
def test_autoscaling_adjustment_types(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_adjustment_types()
    self.assert_request_parameters({'Action': 'DescribeAdjustmentTypes'}, ignore_params_values=['Version'])
    self.assertTrue(isinstance(response, list))
    self.assertEqual(response[0].adjustment_type, 'ChangeInCapacity')
    self.assertEqual(response[1].adjustment_type, 'ExactCapacity')
    self.assertEqual(response[2].adjustment_type, 'PercentChangeInCapacity')