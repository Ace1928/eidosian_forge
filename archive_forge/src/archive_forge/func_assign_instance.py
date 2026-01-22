import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def assign_instance(self, instance_id, layer_ids):
    """
        Assign a registered instance to a custom layer. You cannot use
        this action with instances that were created with AWS
        OpsWorks.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type instance_id: string
        :param instance_id: The instance ID.

        :type layer_ids: list
        :param layer_ids: The layer ID, which must correspond to a custom
            layer. You cannot assign a registered instance to a built-in layer.

        """
    params = {'InstanceId': instance_id, 'LayerIds': layer_ids}
    return self.make_request(action='AssignInstance', body=json.dumps(params))