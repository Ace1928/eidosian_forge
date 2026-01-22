import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.logs import exceptions
from boto.compat import json
def create_log_stream(self, log_group_name, log_stream_name):
    """
        Creates a new log stream in the specified log group. The name
        of the log stream must be unique within the log group. There
        is no limit on the number of log streams that can exist in a
        log group.

        You must use the following guidelines when naming a log
        stream:

        + Log stream names can be between 1 and 512 characters long.
        + The ':' colon character is not allowed.

        :type log_group_name: string
        :param log_group_name:

        :type log_stream_name: string
        :param log_stream_name:

        """
    params = {'logGroupName': log_group_name, 'logStreamName': log_stream_name}
    return self.make_request(action='CreateLogStream', body=json.dumps(params))