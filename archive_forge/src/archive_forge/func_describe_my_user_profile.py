import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def describe_my_user_profile(self):
    """
        Describes a user's SSH information.

        **Required Permissions**: To use this action, an IAM user must
        have self-management enabled or an attached policy that
        explicitly grants permissions. For more information on user
        permissions, see `Managing User Permissions`_.

        
        """
    params = {}
    return self.make_request(action='DescribeMyUserProfile', body=json.dumps(params))