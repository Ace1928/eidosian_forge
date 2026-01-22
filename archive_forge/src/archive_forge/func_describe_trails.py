import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def describe_trails(self, trail_name_list=None):
    """
        Retrieves settings for the trail associated with the current
        region for your account.

        :type trail_name_list: list
        :param trail_name_list: The trail returned.

        """
    params = {}
    if trail_name_list is not None:
        params['trailNameList'] = trail_name_list
    return self.make_request(action='DescribeTrails', body=json.dumps(params))