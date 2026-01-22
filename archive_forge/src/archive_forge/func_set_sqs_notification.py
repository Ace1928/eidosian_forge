import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def set_sqs_notification(self, hit_type, queue_url, event_types=None):
    """
        Performs a SetHITTypeNotification operation so set SQS notification
        for a specified HIT type. Queue URL is of form:
        https://queue.amazonaws.com/<CUSTOMER_ID>/<QUEUE_NAME> and can be
        found when looking at the details for a Queue in the AWS Console
        """
    return self._set_notification(hit_type, 'SQS', queue_url, 'SetHITTypeNotification', event_types)