from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.elastictranscoder import exceptions
def list_jobs_by_pipeline(self, pipeline_id=None, ascending=None, page_token=None):
    """
        The ListJobsByPipeline operation gets a list of the jobs
        currently in a pipeline.

        Elastic Transcoder returns all of the jobs currently in the
        specified pipeline. The response body contains one element for
        each job that satisfies the search criteria.

        :type pipeline_id: string
        :param pipeline_id: The ID of the pipeline for which you want to get
            job information.

        :type ascending: string
        :param ascending: To list jobs in chronological order by the date and
            time that they were submitted, enter `True`. To list jobs in
            reverse chronological order, enter `False`.

        :type page_token: string
        :param page_token: When Elastic Transcoder returns more than one page
            of results, use `pageToken` in subsequent `GET` requests to get
            each successive page of results.

        """
    uri = '/2012-09-25/jobsByPipeline/{0}'.format(pipeline_id)
    params = {}
    if pipeline_id is not None:
        params['PipelineId'] = pipeline_id
    if ascending is not None:
        params['Ascending'] = ascending
    if page_token is not None:
        params['PageToken'] = page_token
    return self.make_request('GET', uri, expected_status=200, params=params)