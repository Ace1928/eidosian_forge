import time
from tests.unit import unittest
from boto.datapipeline import layer1
def test_validate_pipeline(self):
    pipeline_id = self.create_pipeline('name2', 'unique_id2')
    self.connection.validate_pipeline_definition(self.sample_pipeline_objects, pipeline_id)