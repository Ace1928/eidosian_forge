import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def attach_group_policy(self, policy_arn, group_name):
    """
        :type policy_arn: string
        :param policy_arn: The ARN of the policy to attach

        :type group_name: string
        :param group_name: Group to attach the policy to

        """
    params = {'PolicyArn': policy_arn, 'GroupName': group_name}
    return self.get_response('AttachGroupPolicy', params)