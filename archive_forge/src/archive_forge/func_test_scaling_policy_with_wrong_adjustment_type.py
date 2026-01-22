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
def test_scaling_policy_with_wrong_adjustment_type(self):
    self.set_http_response(status_code=200)
    policy = ScalingPolicy(name='foo', as_name='bar', adjustment_type='ChangeInCapacity', scaling_adjustment=50, min_adjustment_step=30)
    self.service_connection.create_scaling_policy(policy)
    self.assert_request_parameters({'Action': 'PutScalingPolicy', 'PolicyName': 'foo', 'AutoScalingGroupName': 'bar', 'AdjustmentType': 'ChangeInCapacity', 'ScalingAdjustment': 50}, ignore_params_values=['Version'])