import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_stack_summary(self, stack_id):
    """
        Describes the number of layers and apps in a specified stack,
        and the number of instances in each state, such as
        `running_setup` or `online`.

        **Required Permissions**: To use this action, an IAM user must
        have a Show, Deploy, or Manage permissions level for the
        stack, or an attached policy that explicitly grants
        permissions. For more information on user permissions, see
        `Managing User Permissions`_.

        :type stack_id: string
        :param stack_id: The stack ID.

        """
    params = {'StackId': stack_id}
    return self.make_request(action='DescribeStackSummary', body=json.dumps(params))