import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudtrail import exceptions
from boto.compat import json
def delete_trail(self, name):
    """
        Deletes a trail.

        :type name: string
        :param name: The name of a trail to be deleted.

        """
    params = {'Name': name}
    return self.make_request(action='DeleteTrail', body=json.dumps(params))