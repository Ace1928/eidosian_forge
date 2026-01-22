import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def start_logging(self, name):
    """
        Starts the recording of AWS API calls and log file delivery
        for a trail.

        :type name: string
        :param name: The name of the trail for which CloudTrail logs AWS API
            calls.

        """
    params = {'Name': name}
    return self.make_request(action='StartLogging', body=json.dumps(params))