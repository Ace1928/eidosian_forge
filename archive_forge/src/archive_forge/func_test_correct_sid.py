import hashlib
import time
from tests.unit import unittest
from boto.compat import json
from boto.sqs.connection import SQSConnection
from boto.sns.connection import SNSConnection
def test_correct_sid(self):
    now = time.time()
    topic_name = queue_name = 'test_correct_sid%d' % now
    timeout = 60
    queue = self.sqsc.create_queue(queue_name, timeout)
    self.addCleanup(self.sqsc.delete_queue, queue, True)
    queue_arn = queue.arn
    topic = self.snsc.create_topic(topic_name)
    topic_arn = topic['CreateTopicResponse']['CreateTopicResult']['TopicArn']
    self.addCleanup(self.snsc.delete_topic, topic_arn)
    expected_sid = hashlib.md5((topic_arn + queue_arn).encode('utf-8')).hexdigest()
    resp = self.snsc.subscribe_sqs_queue(topic_arn, queue)
    found_expected_sid = False
    statements = self.get_policy_statements(queue)
    for statement in statements:
        if statement['Sid'] == expected_sid:
            found_expected_sid = True
            break
    self.assertTrue(found_expected_sid)