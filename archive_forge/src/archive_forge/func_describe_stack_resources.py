import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def describe_stack_resources(self, stack_name_or_id=None, logical_resource_id=None, physical_resource_id=None):
    """
        Returns AWS resource descriptions for running and deleted
        stacks. If `StackName` is specified, all the associated
        resources that are part of the stack are returned. If
        `PhysicalResourceId` is specified, the associated resources of
        the stack that the resource belongs to are returned.
        Only the first 100 resources will be returned. If your stack
        has more resources than this, you should use
        `ListStackResources` instead.
        For deleted stacks, `DescribeStackResources` returns resource
        information for up to 90 days after the stack has been
        deleted.

        You must specify either `StackName` or `PhysicalResourceId`,
        but not both. In addition, you can specify `LogicalResourceId`
        to filter the returned result. For more information about
        resources, the `LogicalResourceId` and `PhysicalResourceId`,
        go to the `AWS CloudFormation User Guide`_.
        A `ValidationError` is returned if you specify both
        `StackName` and `PhysicalResourceId` in the same request.

        :type stack_name_or_id: string
        :param stack_name_or_id: The name or the unique identifier associated
            with the stack.
        Required: Conditional. If you do not specify `StackName`, you must
            specify `PhysicalResourceId`.

        Default: There is no default value.

        :type logical_resource_id: string
        :param logical_resource_id: The logical name of the resource as
            specified in the template.
        Default: There is no default value.

        :type physical_resource_id: string
        :param physical_resource_id: The name or unique identifier that
            corresponds to a physical instance ID of a resource supported by
            AWS CloudFormation.
        For example, for an Amazon Elastic Compute Cloud (EC2) instance,
            `PhysicalResourceId` corresponds to the `InstanceId`. You can pass
            the EC2 `InstanceId` to `DescribeStackResources` to find which
            stack the instance belongs to and what other resources are part of
            the stack.

        Required: Conditional. If you do not specify `PhysicalResourceId`, you
            must specify `StackName`.

        Default: There is no default value.

        """
    params = {}
    if stack_name_or_id:
        params['StackName'] = stack_name_or_id
    if logical_resource_id:
        params['LogicalResourceId'] = logical_resource_id
    if physical_resource_id:
        params['PhysicalResourceId'] = physical_resource_id
    return self.get_list('DescribeStackResources', params, [('member', StackResource)])