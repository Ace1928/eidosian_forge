import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def register_volume(self, stack_id, ec_2_volume_id=None):
    """
        Registers an Amazon EBS volume with a specified stack. A
        volume can be registered with only one stack at a time. If the
        volume is already registered, you must first deregister it by
        calling DeregisterVolume. For more information, see `Resource
        Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type ec_2_volume_id: string
        :param ec_2_volume_id: The Amazon EBS volume ID.

        :type stack_id: string
        :param stack_id: The stack ID.

        """
    params = {'StackId': stack_id}
    if ec_2_volume_id is not None:
        params['Ec2VolumeId'] = ec_2_volume_id
    return self.make_request(action='RegisterVolume', body=json.dumps(params))