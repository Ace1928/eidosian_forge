import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def delete_realtime_endpoint(self, ml_model_id):
    """
        Deletes a real time endpoint of an `MLModel`.

        :type ml_model_id: string
        :param ml_model_id: The ID assigned to the `MLModel` during creation.

        """
    params = {'MLModelId': ml_model_id}
    return self.make_request(action='DeleteRealtimeEndpoint', body=json.dumps(params))