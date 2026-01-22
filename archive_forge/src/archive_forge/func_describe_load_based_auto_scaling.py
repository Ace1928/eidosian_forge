import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_load_based_auto_scaling(self, layer_ids):
    """
        Describes load-based auto scaling configurations for specified
        layers.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type layer_ids: list
        :param layer_ids: An array of layer IDs.

        """
    params = {'LayerIds': layer_ids}
    return self.make_request(action='DescribeLoadBasedAutoScaling', body=json.dumps(params))