import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def put_pipeline_definition(self, pipeline_objects, pipeline_id):
    """
        Adds tasks, schedules, and preconditions that control the
        behavior of the pipeline. You can use PutPipelineDefinition to
        populate a new pipeline or to update an existing pipeline that
        has not yet been activated.

        PutPipelineDefinition also validates the configuration as it
        adds it to the pipeline. Changes to the pipeline are saved
        unless one of the following three validation errors exists in
        the pipeline.

        #. An object is missing a name or identifier field.
        #. A string or reference field is empty.
        #. The number of objects in the pipeline exceeds the maximum
           allowed objects.



        Pipeline object definitions are passed to the
        PutPipelineDefinition action and returned by the
        GetPipelineDefinition action.

        :type pipeline_id: string
        :param pipeline_id: The identifier of the pipeline to be configured.

        :type pipeline_objects: list
        :param pipeline_objects: The objects that define the pipeline. These
            will overwrite the existing pipeline definition.

        """
    params = {'pipelineId': pipeline_id, 'pipelineObjects': pipeline_objects}
    return self.make_request(action='PutPipelineDefinition', body=json.dumps(params))