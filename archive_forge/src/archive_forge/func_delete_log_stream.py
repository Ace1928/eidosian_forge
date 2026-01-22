import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def delete_log_stream(self, log_group_name, log_stream_name):
    """
        Deletes a log stream and permanently deletes all the archived
        log events associated with it.

        :type log_group_name: string
        :param log_group_name:

        :type log_stream_name: string
        :param log_stream_name:

        """
    params = {'logGroupName': log_group_name, 'logStreamName': log_stream_name}
    return self.make_request(action='DeleteLogStream', body=json.dumps(params))