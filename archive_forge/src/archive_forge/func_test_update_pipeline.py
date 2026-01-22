import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
def test_update_pipeline(self):
    pipeline_id = self.create_pipeline()
    self.api.update_pipeline_status(pipeline_id, 'Paused')
    response = self.api.read_pipeline(pipeline_id)
    self.assertEqual(response['Pipeline']['Status'], 'Paused')