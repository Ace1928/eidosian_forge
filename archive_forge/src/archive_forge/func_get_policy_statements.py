import hashlib
import time
from tests.unit import unittest
from boto.compat import json
from boto.sqs.connection import SQSConnection
from boto.sns.connection import SNSConnection
def get_policy_statements(self, queue):
    attrs = queue.get_attributes('Policy')
    policy = json.loads(attrs.get('Policy', '{}'))
    return policy.get('Statement', {})