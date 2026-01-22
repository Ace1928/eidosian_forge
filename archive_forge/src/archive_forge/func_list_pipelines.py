import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def list_pipelines(self, marker=None):
    """
        Returns a list of pipeline identifiers for all active
        pipelines. Identifiers are returned only for pipelines you
        have permission to access.

        :type marker: string
        :param marker: The starting point for the results to be returned. The
            first time you call ListPipelines, this value should be empty. As
            long as the action returns `HasMoreResults` as `True`, you can call
            ListPipelines again and pass the marker value from the response to
            retrieve the next set of results.

        """
    params = {}
    if marker is not None:
        params['marker'] = marker
    return self.make_request(action='ListPipelines', body=json.dumps(params))