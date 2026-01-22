import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
def test_update_pipeline_notification(self):
    pipeline_id = self.create_pipeline()
    response = self.sns.create_topic('pipeline-errors')
    topic_arn = response['CreateTopicResponse']['CreateTopicResult']['TopicArn']
    self.addCleanup(self.sns.delete_topic, topic_arn)
    self.api.update_pipeline_notifications(pipeline_id, {'Progressing': '', 'Completed': '', 'Warning': '', 'Error': topic_arn})
    response = self.api.read_pipeline(pipeline_id)
    self.assertEqual(response['Pipeline']['Notifications']['Error'], topic_arn)