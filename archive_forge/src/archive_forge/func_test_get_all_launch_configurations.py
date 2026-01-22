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
def test_get_all_launch_configurations(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_launch_configurations()
    self.assertTrue(isinstance(response, list))
    self.assertEqual(len(response), 1)
    self.assertTrue(isinstance(response[0], LaunchConfiguration))
    self.assertEqual(response[0].associate_public_ip_address, True)
    self.assertEqual(response[0].name, 'my-test-lc')
    self.assertEqual(response[0].instance_type, 'm1.small')
    self.assertEqual(response[0].launch_configuration_arn, 'arn:aws:autoscaling:us-east-1:803981987763:launchConfiguration:9dbbbf87-6141-428a-a409-0752edbe6cad:launchConfigurationName/my-test-lc')
    self.assertEqual(response[0].image_id, 'ami-514ac838')
    self.assertTrue(isinstance(response[0].instance_monitoring, launchconfig.InstanceMonitoring))
    self.assertEqual(response[0].instance_monitoring.enabled, 'true')
    self.assertEqual(response[0].ebs_optimized, False)
    self.assertEqual(response[0].block_device_mappings, [])
    self.assertEqual(response[0].classic_link_vpc_id, 'vpc-12345')
    self.assertEqual(response[0].classic_link_vpc_security_groups, ['sg-1234'])
    self.assert_request_parameters({'Action': 'DescribeLaunchConfigurations'}, ignore_params_values=['Version'])