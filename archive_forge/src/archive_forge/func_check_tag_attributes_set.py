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
def check_tag_attributes_set(self, name, value, attr):
    tag = Tag()
    tag.endElement(name, value, None)
    if value == 'true':
        self.assertEqual(getattr(tag, attr), True)
    else:
        self.assertEqual(getattr(tag, attr), value)