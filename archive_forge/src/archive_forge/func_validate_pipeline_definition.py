import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.datapipeline import exceptions
def validate_pipeline_definition(self, pipeline_objects, pipeline_id):
    """
        Tests the pipeline definition with a set of validation checks
        to ensure that it is well formed and can run without error.

        :type pipeline_id: string
        :param pipeline_id: Identifies the pipeline whose definition is to be
            validated.

        :type pipeline_objects: list
        :param pipeline_objects: A list of objects that define the pipeline
            changes to validate against the pipeline.

        """
    params = {'pipelineId': pipeline_id, 'pipelineObjects': pipeline_objects}
    return self.make_request(action='ValidatePipelineDefinition', body=json.dumps(params))