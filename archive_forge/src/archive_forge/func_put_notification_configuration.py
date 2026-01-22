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
def put_notification_configuration(self, autoscale_group, topic, notification_types):
    """
        Configures an Auto Scaling group to send notifications when
        specified events take place.

        :type autoscale_group: str or
            :class:`boto.ec2.autoscale.group.AutoScalingGroup` object
        :param autoscale_group: The Auto Scaling group to put notification
            configuration on.

        :type topic: str
        :param topic: The Amazon Resource Name (ARN) of the Amazon Simple
            Notification Service (SNS) topic.

        :type notification_types: list
        :param notification_types: The type of events that will trigger
            the notification. Valid types are:
            'autoscaling:EC2_INSTANCE_LAUNCH',
            'autoscaling:EC2_INSTANCE_LAUNCH_ERROR',
            'autoscaling:EC2_INSTANCE_TERMINATE',
            'autoscaling:EC2_INSTANCE_TERMINATE_ERROR',
            'autoscaling:TEST_NOTIFICATION'
        """
    name = autoscale_group
    if isinstance(autoscale_group, AutoScalingGroup):
        name = autoscale_group.name
    params = {'AutoScalingGroupName': name, 'TopicARN': topic}
    self.build_list_params(params, notification_types, 'NotificationTypes')
    return self.get_status('PutNotificationConfiguration', params)