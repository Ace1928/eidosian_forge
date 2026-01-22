import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_group_policy(self, group_name, policy_name):
    """
        Retrieves the specified policy document for the specified group.

        :type group_name: string
        :param group_name: The name of the group the policy is associated with.

        :type policy_name: string
        :param policy_name: The policy document to get.

        """
    params = {'GroupName': group_name, 'PolicyName': policy_name}
    return self.get_response('GetGroupPolicy', params, verb='POST')