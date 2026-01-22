import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def update_batch_prediction(self, batch_prediction_id, batch_prediction_name):
    """
        Updates the `BatchPredictionName` of a `BatchPrediction`.

        You can use the GetBatchPrediction operation to view the
        contents of the updated data element.

        :type batch_prediction_id: string
        :param batch_prediction_id: The ID assigned to the `BatchPrediction`
            during creation.

        :type batch_prediction_name: string
        :param batch_prediction_name: A new user-supplied name or description
            of the `BatchPrediction`.

        """
    params = {'BatchPredictionId': batch_prediction_id, 'BatchPredictionName': batch_prediction_name}
    return self.make_request(action='UpdateBatchPrediction', body=json.dumps(params))