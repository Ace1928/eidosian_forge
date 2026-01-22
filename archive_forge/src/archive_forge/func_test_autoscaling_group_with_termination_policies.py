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
def test_autoscaling_group_with_termination_policies(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_termination_policies()
    self.assertListEqual(response, ['ClosestToNextInstanceHour', 'Default', 'NewestInstance', 'OldestInstance', 'OldestLaunchConfiguration'])