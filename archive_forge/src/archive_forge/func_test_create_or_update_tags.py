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
def test_create_or_update_tags(self):
    self.set_http_response(status_code=200)
    tags = [Tag(connection=self.service_connection, key='alpha', value='tango', resource_id='sg-00000000', resource_type='auto-scaling-group', propagate_at_launch=True), Tag(connection=self.service_connection, key='bravo', value='sierra', resource_id='sg-00000000', resource_type='auto-scaling-group', propagate_at_launch=False)]
    response = self.service_connection.create_or_update_tags(tags)
    self.assert_request_parameters({'Action': 'CreateOrUpdateTags', 'Tags.member.1.ResourceType': 'auto-scaling-group', 'Tags.member.1.ResourceId': 'sg-00000000', 'Tags.member.1.Key': 'alpha', 'Tags.member.1.Value': 'tango', 'Tags.member.1.PropagateAtLaunch': 'true', 'Tags.member.2.ResourceType': 'auto-scaling-group', 'Tags.member.2.ResourceId': 'sg-00000000', 'Tags.member.2.Key': 'bravo', 'Tags.member.2.Value': 'sierra', 'Tags.member.2.PropagateAtLaunch': 'false'}, ignore_params_values=['Version'])