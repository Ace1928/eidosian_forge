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
def set_email_notification(self, hit_type, email, event_types=None):
    """
        Performs a SetHITTypeNotification operation to set email
        notification for a specified HIT type
        """
    return self._set_notification(hit_type, 'Email', email, 'SetHITTypeNotification', event_types)