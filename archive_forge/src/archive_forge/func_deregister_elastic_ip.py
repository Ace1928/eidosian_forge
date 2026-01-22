import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def deregister_elastic_ip(self, elastic_ip):
    """
        Deregisters a specified Elastic IP address. The address can
        then be registered by another stack. For more information, see
        `Resource Management`_.

        **Required Permissions**: To use this action, an IAM user must
        have a Manage permissions level for the stack, or an attached
        policy that explicitly grants permissions. For more
        information on user permissions, see `Managing User
        Permissions`_.

        :type elastic_ip: string
        :param elastic_ip: The Elastic IP address.

        """
    params = {'ElasticIp': elastic_ip}
    return self.make_request(action='DeregisterElasticIp', body=json.dumps(params))