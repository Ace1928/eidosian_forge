import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudhsm import exceptions
def create_hapg(self, label):
    """
        Creates a high-availability partition group. A high-
        availability partition group is a group of partitions that
        spans multiple physical HSMs.

        :type label: string
        :param label: The label of the new high-availability partition group.

        """
    params = {'Label': label}
    return self.make_request(action='CreateHapg', body=json.dumps(params))