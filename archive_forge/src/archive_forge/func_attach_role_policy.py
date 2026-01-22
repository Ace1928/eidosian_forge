import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def attach_role_policy(self, policy_arn, role_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to attach

        :type role_name: string
        :param role_name: Role to attach the policy to

        """
    params = {'PolicyArn': policy_arn, 'RoleName': role_name}
    return self.get_response('AttachRolePolicy', params)