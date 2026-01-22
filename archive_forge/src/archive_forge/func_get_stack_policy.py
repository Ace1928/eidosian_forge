import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def get_stack_policy(self, stack_name_or_id):
    """
        Returns the stack policy for a specified stack. If a stack
        doesn't have a policy, a null value is returned.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or stack ID that is associated with
            the stack whose policy you want to get.

        :rtype: string
        :return: The policy JSON document
        """
    params = {'ContentType': 'JSON', 'StackName': stack_name_or_id}
    response = self._do_request('GetStackPolicy', params, '/', 'POST')
    return response['GetStackPolicyResponse']['GetStackPolicyResult']['StackPolicyBody']