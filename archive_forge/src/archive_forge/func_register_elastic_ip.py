import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def register_elastic_ip(self, elastic_ip, stack_id):
    """
        Registers an Elastic IP address with a specified stack. An
        address can be registered with only one stack at a time. If
        the address is already registered, you must first deregister
        it by calling DeregisterElasticIp. For more information, see
        `Resource Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type elastic_ip: string
        :param elastic_ip: The Elastic IP address.

        :type stack_id: string
        :param stack_id: The stack ID.

        """
    params = {'ElasticIp': elastic_ip, 'StackId': stack_id}
    return self.make_request(action='RegisterElasticIp', body=json.dumps(params))