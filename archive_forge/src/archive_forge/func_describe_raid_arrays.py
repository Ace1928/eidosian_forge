import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_raid_arrays(self, instance_id=None, stack_id=None, raid_array_ids=None):
    """
        Describe an instance's RAID arrays.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID. If you use this parameter,
            `DescribeRaidArrays` returns descriptions of the RAID arrays
            associated with the specified instance.

        :type stack_id: string
        :param stack_id: The stack ID.

        :type raid_array_ids: list
        :param raid_array_ids: An array of RAID array IDs. If you use this
            parameter, `DescribeRaidArrays` returns descriptions of the
            specified arrays. Otherwise, it returns a description of every
            array.

        """
    params = {}
    if instance_id is not None:
        params['InstanceId'] = instance_id
    if stack_id is not None:
        params['StackId'] = stack_id
    if raid_array_ids is not None:
        params['RaidArrayIds'] = raid_array_ids
    return self.make_request(action='DescribeRaidArrays', body=json.dumps(params))