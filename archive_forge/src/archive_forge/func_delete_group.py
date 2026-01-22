import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def delete_group(self, group_name):
    """
        Delete a group. The group must not contain any Users or
        have any attached policies

        :type group_name: string
        :param group_name: The name of the group to delete.

        """
    params = {'GroupName': group_name}
    return self.get_response('DeleteGroup', params)