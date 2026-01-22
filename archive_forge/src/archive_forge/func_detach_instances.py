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
def detach_instances(self, name, instance_ids, decrement_capacity=True):
    """
        Detach instances from an Auto Scaling group.

        :type name: str
        :param name: The name of the Auto Scaling group from which to detach instances.

        :type instance_ids: list
        :param instance_ids: Instance ids to be detached from the Auto Scaling group.

        :type decrement_capacity: bool
        :param decrement_capacity: Whether to decrement the size of the
            Auto Scaling group or not.
        """
    params = {'AutoScalingGroupName': name}
    params['ShouldDecrementDesiredCapacity'] = 'true' if decrement_capacity else 'false'
    self.build_list_params(params, instance_ids, 'InstanceIds')
    return self.get_status('DetachInstances', params)