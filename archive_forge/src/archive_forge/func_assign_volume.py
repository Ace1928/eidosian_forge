import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def assign_volume(self, volume_id, instance_id=None):
    """
        Assigns one of the stack's registered Amazon EBS volumes to a
        specified instance. The volume must first be registered with
        the stack by calling RegisterVolume. For more information, see
        `Resource Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type volume_id: string
        :param volume_id: The volume ID.

        :type instance_id: string
        :param instance_id: The instance ID.

        """
    params = {'VolumeId': volume_id}
    if instance_id is not None:
        params['InstanceId'] = instance_id
    return self.make_request(action='AssignVolume', body=json.dumps(params))