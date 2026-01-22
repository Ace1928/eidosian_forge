import time
from tests.unit import unittest
from boto.datapipeline import layer1
def create_pipeline(self, name, unique_id, description=None):
    response = self.connection.create_pipeline(name, unique_id, description)
    pipeline_id = response['pipelineId']
    self.addCleanup(self.connection.delete_pipeline, pipeline_id)
    return pipeline_id