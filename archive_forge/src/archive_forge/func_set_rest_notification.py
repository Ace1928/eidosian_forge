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
def set_rest_notification(self, hit_type, url, event_types=None):
    """
        Performs a SetHITTypeNotification operation to set REST notification
        for a specified HIT type
        """
    return self._set_notification(hit_type, 'REST', url, 'SetHITTypeNotification', event_types)