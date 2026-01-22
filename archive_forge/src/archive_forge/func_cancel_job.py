from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def cancel_job(self, id=None):
    """
        The CancelJob operation cancels an unfinished job.
        You can only cancel a job that has a status of `Submitted`. To
        prevent a pipeline from starting to process a job while you're
        getting the job identifier, use UpdatePipelineStatus to
        temporarily pause the pipeline.

        :type id: string
        :param id: The identifier of the job that you want to cancel.
        To get a list of the jobs (including their `jobId`) that have a status
            of `Submitted`, use the ListJobsByStatus API action.

        """
    uri = '/2012-09-25/jobs/{0}'.format(id)
    return self.make_request('DELETE', uri, expected_status=202)