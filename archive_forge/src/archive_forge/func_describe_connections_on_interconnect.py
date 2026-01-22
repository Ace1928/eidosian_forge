import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.directconnect import exceptions
from boto.compat import json
def describe_connections_on_interconnect(self, interconnect_id):
    """
        Return a list of connections that have been provisioned on the
        given interconnect.

        :type interconnect_id: string
        :param interconnect_id: ID of the interconnect on which a list of
            connection is provisioned.
        Example: dxcon-abc123

        Default: None

        """
    params = {'interconnectId': interconnect_id}
    return self.make_request(action='DescribeConnectionsOnInterconnect', body=json.dumps(params))