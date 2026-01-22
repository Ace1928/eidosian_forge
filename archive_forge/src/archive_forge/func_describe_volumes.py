import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_volumes(self, instance_id=None, stack_id=None, raid_array_id=None, volume_ids=None):
    """
        Describes an instance's Amazon EBS volumes.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID. If you use this parameter,
            `DescribeVolumes` returns descriptions of the volumes associated
            with the specified instance.

        :type stack_id: string
        :param stack_id: A stack ID. The action describes the stack's
            registered Amazon EBS volumes.

        :type raid_array_id: string
        :param raid_array_id: The RAID array ID. If you use this parameter,
            `DescribeVolumes` returns descriptions of the volumes associated
            with the specified RAID array.

        :type volume_ids: list
        :param volume_ids: Am array of volume IDs. If you use this parameter,
            `DescribeVolumes` returns descriptions of the specified volumes.
            Otherwise, it returns a description of every volume.

        """
    params = {}
    if instance_id is not None:
        params['InstanceId'] = instance_id
    if stack_id is not None:
        params['StackId'] = stack_id
    if raid_array_id is not None:
        params['RaidArrayId'] = raid_array_id
    if volume_ids is not None:
        params['VolumeIds'] = volume_ids
    return self.make_request(action='DescribeVolumes', body=json.dumps(params))