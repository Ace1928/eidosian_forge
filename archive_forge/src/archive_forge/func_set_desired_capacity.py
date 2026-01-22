import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo, get_regions, load_regions
from boto.regioninfo import connect
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.group import AutoScalingGroup
from boto.ec2.autoscale.group import ProcessType
from boto.ec2.autoscale.activity import Activity
from boto.ec2.autoscale.policy import AdjustmentType
from boto.ec2.autoscale.policy import MetricCollectionTypes
from boto.ec2.autoscale.policy import ScalingPolicy
from boto.ec2.autoscale.policy import TerminationPolicies
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.scheduled import ScheduledUpdateGroupAction
from boto.ec2.autoscale.tag import Tag
from boto.ec2.autoscale.limits import AccountLimits
from boto.compat import six
def set_desired_capacity(self, group_name, desired_capacity, honor_cooldown=False):
    """
        Adjusts the desired size of the AutoScalingGroup by initiating scaling
        activities. When reducing the size of the group, it is not possible to define
        which Amazon EC2 instances will be terminated. This applies to any Auto Scaling
        decisions that might result in terminating instances.

        :type group_name: string
        :param group_name: name of the auto scaling group

        :type desired_capacity: integer
        :param desired_capacity: new capacity setting for auto scaling group

        :type honor_cooldown: boolean
        :param honor_cooldown: by default, overrides any cooldown period
        """
    params = {'AutoScalingGroupName': group_name, 'DesiredCapacity': desired_capacity}
    if honor_cooldown:
        params['HonorCooldown'] = 'true'
    return self.get_status('SetDesiredCapacity', params)