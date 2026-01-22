import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_role_policy(self, role_name, policy_name):
    """
        Deletes the specified policy associated with the specified role.

        :type role_name: string
        :param role_name: Name of the role associated with the policy.

        :type policy_name: string
        :param policy_name: Name of the policy to delete.
        """
    return self.get_response('DeleteRolePolicy', {'RoleName': role_name, 'PolicyName': policy_name})