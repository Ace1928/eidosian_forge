import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def cancel_update_stack(self, stack_name_or_id=None):
    """
        Cancels an update on the specified stack. If the call
        completes successfully, the stack will roll back the update
        and revert to the previous stack configuration.
        Only stacks that are in the UPDATE_IN_PROGRESS state can be
        canceled.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated with
            the stack.

        """
    params = {}
    if stack_name_or_id:
        params['StackName'] = stack_name_or_id
    return self.get_status('CancelUpdateStack', params)