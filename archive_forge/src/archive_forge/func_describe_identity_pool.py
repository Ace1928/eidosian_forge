import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def describe_identity_pool(self, identity_pool_id):
    """
        Gets details about a particular identity pool, including the
        pool name, ID description, creation date, and current number
        of users.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        """
    params = {'IdentityPoolId': identity_pool_id}
    return self.make_request(action='DescribeIdentityPool', body=json.dumps(params))