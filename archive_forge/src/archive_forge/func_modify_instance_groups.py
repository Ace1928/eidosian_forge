import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def modify_instance_groups(self, instance_group_ids, new_sizes):
    """
        Modify the number of nodes and configuration settings in an
        instance group.

        :type instance_group_ids: list(str)
        :param instance_group_ids: A list of the ID's of the instance
            groups to be modified

        :type new_sizes: list(int)
        :param new_sizes: A list of the new sizes for each instance group
        """
    if not isinstance(instance_group_ids, list):
        instance_group_ids = [instance_group_ids]
    if not isinstance(new_sizes, list):
        new_sizes = [new_sizes]
    instance_groups = zip(instance_group_ids, new_sizes)
    params = {}
    for k, ig in enumerate(instance_groups):
        params['InstanceGroups.member.%d.InstanceGroupId' % (k + 1)] = ig[0]
        params['InstanceGroups.member.%d.InstanceCount' % (k + 1)] = ig[1]
    return self.get_object('ModifyInstanceGroups', params, ModifyInstanceGroupsResponse, verb='POST')