import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def delete_retention_policy(self, log_group_name):
    """
        

        :type log_group_name: string
        :param log_group_name:

        """
    params = {'logGroupName': log_group_name}
    return self.make_request(action='DeleteRetentionPolicy', body=json.dumps(params))