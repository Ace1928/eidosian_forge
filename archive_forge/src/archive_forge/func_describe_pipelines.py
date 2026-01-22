import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def describe_pipelines(self, pipeline_ids):
    """
        Retrieve metadata about one or more pipelines. The information
        retrieved includes the name of the pipeline, the pipeline
        identifier, its current state, and the user account that owns
        the pipeline. Using account credentials, you can retrieve
        metadata about pipelines that you or your IAM users have
        created. If you are using an IAM user account, you can
        retrieve metadata about only those pipelines you have read
        permission for.

        To retrieve the full pipeline definition instead of metadata
        about the pipeline, call the GetPipelineDefinition action.

        :type pipeline_ids: list
        :param pipeline_ids: Identifiers of the pipelines to describe. You can
            pass as many as 25 identifiers in a single call to
            DescribePipelines. You can obtain pipeline identifiers by calling
            ListPipelines.

        """
    params = {'pipelineIds': pipeline_ids}
    return self.make_request(action='DescribePipelines', body=json.dumps(params))