import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_layers(self, stack_id=None, layer_ids=None):
    """
        Requests a description of one or more layers in a specified
        stack.


        You must specify at least one of the parameters.


        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        :type layer_ids: list
        :param layer_ids: An array of layer IDs that specify the layers to be
            described. If you omit this parameter, `DescribeLayers` returns a
            description of every layer in the specified stack.

        """
    params = {}
    if stack_id is not None:
        params['StackId'] = stack_id
    if layer_ids is not None:
        params['LayerIds'] = layer_ids
    return self.make_request(action='DescribeLayers', body=json.dumps(params))