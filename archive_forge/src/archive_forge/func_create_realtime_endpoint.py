import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_realtime_endpoint(self, ml_model_id):
    """
        Creates a real-time endpoint for the `MLModel`. The endpoint
        contains the URI of the `MLModel`; that is, the location to
        send real-time prediction requests for the specified
        `MLModel`.

        :type ml_model_id: string
        :param ml_model_id: The ID assigned to the `MLModel` during creation.

        """
    params = {'MLModelId': ml_model_id}
    return self.make_request(action='CreateRealtimeEndpoint', body=json.dumps(params))