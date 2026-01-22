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
def enable_metrics_collection(self, as_group, granularity, metrics=None):
    """
        Enables monitoring of group metrics for the Auto Scaling group
        specified in AutoScalingGroupName. You can specify the list of enabled
        metrics with the Metrics parameter.

        Auto scaling metrics collection can be turned on only if the
        InstanceMonitoring.Enabled flag, in the Auto Scaling group's launch
        configuration, is set to true.

        :type autoscale_group: string
        :param autoscale_group: The auto scaling group to get activities on.

        :type granularity: string
        :param granularity: The granularity to associate with the metrics to
            collect. Currently, the only legal granularity is "1Minute".

        :type metrics: string list
        :param metrics: The list of metrics to collect. If no metrics are
                        specified, all metrics are enabled.
        """
    params = {'AutoScalingGroupName': as_group, 'Granularity': granularity}
    if metrics:
        self.build_list_params(params, metrics, 'Metrics')
    return self.get_status('EnableMetricsCollection', params)