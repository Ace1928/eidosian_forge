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
def get_all_groups(self, names=None, max_records=None, next_token=None):
    """
        Returns a full description of each Auto Scaling group in the given
        list. This includes all Amazon EC2 instances that are members of the
        group. If a list of names is not provided, the service returns the full
        details of all Auto Scaling groups.

        This action supports pagination by returning a token if there are more
        pages to retrieve. To get the next page, call this action again with
        the returned token as the NextToken parameter.

        :type names: list
        :param names: List of group names which should be searched for.

        :type max_records: int
        :param max_records: Maximum amount of groups to return.

        :rtype: list
        :returns: List of :class:`boto.ec2.autoscale.group.AutoScalingGroup`
            instances.
        """
    params = {}
    if max_records:
        params['MaxRecords'] = max_records
    if next_token:
        params['NextToken'] = next_token
    if names:
        self.build_list_params(params, names, 'AutoScalingGroupNames')
    return self.get_list('DescribeAutoScalingGroups', params, [('member', AutoScalingGroup)])