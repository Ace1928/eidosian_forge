import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def delete_pipeline(self, pipeline_id):
    """
        Permanently deletes a pipeline, its pipeline definition and
        its run history. You cannot query or restore a deleted
        pipeline. AWS Data Pipeline will attempt to cancel instances
        associated with the pipeline that are currently being
        processed by task runners. Deleting a pipeline cannot be
        undone.

        To temporarily pause a pipeline instead of deleting it, call
        SetStatus with the status set to Pause on individual
        components. Components that are paused by SetStatus can be
        resumed.

        :type pipeline_id: string
        :param pipeline_id: The identifier of the pipeline to be deleted.

        """
    params = {'pipelineId': pipeline_id}
    return self.make_request(action='DeletePipeline', body=json.dumps(params))