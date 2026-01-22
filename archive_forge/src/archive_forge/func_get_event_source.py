import os
from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.awslambda import exceptions
def get_event_source(self, uuid):
    """
        Returns configuration information for the specified event
        source mapping (see AddEventSource).

        This operation requires permission for the
        `lambda:GetEventSource` action.

        :type uuid: string
        :param uuid: The AWS Lambda assigned ID of the event source mapping.

        """
    uri = '/2014-11-13/event-source-mappings/{0}'.format(uuid)
    return self.make_request('GET', uri, expected_status=200)