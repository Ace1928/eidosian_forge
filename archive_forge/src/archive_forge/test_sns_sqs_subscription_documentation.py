import hashlib
import time
from tests.unit import unittest
from boto.compat import json
from boto.sqs.connection import SQSConnection
from boto.sns.connection import SNSConnection

Unit tests for subscribing SQS queues to SNS topics.
