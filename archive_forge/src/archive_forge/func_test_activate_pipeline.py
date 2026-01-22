import time
from tests.unit import unittest
from boto.datapipeline import layer1
def test_activate_pipeline(self):
    pipeline_id = self.create_pipeline('name4', 'unique_id4')
    self.connection.put_pipeline_definition(self.sample_pipeline_objects, pipeline_id)
    self.connection.activate_pipeline(pipeline_id)
    attempts = 0
    state = self.get_pipeline_state(pipeline_id)
    while state != 'SCHEDULED' and attempts < 10:
        time.sleep(10)
        attempts += 1
        state = self.get_pipeline_state(pipeline_id)
        if attempts > 10:
            self.fail('Pipeline did not become scheduled after 10 attempts.')
    objects = self.connection.describe_objects(['Default'], pipeline_id)
    field = objects['pipelineObjects'][0]['fields'][0]
    self.assertDictEqual(field, {'stringValue': 'COMPONENT', 'key': '@sphere'})