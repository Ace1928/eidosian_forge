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
def assign_qualification(self, qualification_type_id, worker_id, value=1, send_notification=True):
    params = {'QualificationTypeId': qualification_type_id, 'WorkerId': worker_id, 'IntegerValue': value, 'SendNotification': send_notification}
    return self._process_request('AssignQualification', params)