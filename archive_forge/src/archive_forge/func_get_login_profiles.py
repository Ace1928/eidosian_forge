import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def get_login_profiles(self, user_name):
    """
        Retrieves the login profile for the specified user.

        :type user_name: string
        :param user_name: The username of the user

        """
    params = {'UserName': user_name}
    return self.get_response('GetLoginProfile', params)