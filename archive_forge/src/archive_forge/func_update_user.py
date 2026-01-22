import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def update_user(self, user_name, new_user_name=None, new_path=None):
    """
        Updates name and/or path of the specified user.

        :type user_name: string
        :param user_name: The name of the user

        :type new_user_name: string
        :param new_user_name: If provided, the username of the user will be
            changed to this username.

        :type new_path: string
        :param new_path: If provided, the path of the user will be
            changed to this path.

        """
    params = {'UserName': user_name}
    if new_user_name:
        params['NewUserName'] = new_user_name
    if new_path:
        params['NewPath'] = new_path
    return self.get_response('UpdateUser', params)