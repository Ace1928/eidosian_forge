from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def update_pipeline_status(self, id=None, status=None):
    """
        The UpdatePipelineStatus operation pauses or reactivates a
        pipeline, so that the pipeline stops or restarts the
        processing of jobs.

        Changing the pipeline status is useful if you want to cancel
        one or more jobs. You can't cancel jobs after Elastic
        Transcoder has started processing them; if you pause the
        pipeline to which you submitted the jobs, you have more time
        to get the job IDs for the jobs that you want to cancel, and
        to send a CancelJob request.

        :type id: string
        :param id: The identifier of the pipeline to update.

        :type status: string
        :param status:
        The desired status of the pipeline:


        + `Active`: The pipeline is processing jobs.
        + `Paused`: The pipeline is not currently processing jobs.

        """
    uri = '/2012-09-25/pipelines/{0}/status'.format(id)
    params = {}
    if id is not None:
        params['Id'] = id
    if status is not None:
        params['Status'] = status
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params))