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
def get_qualification_type(self, qualification_type_id):
    params = {'QualificationTypeId': qualification_type_id}
    return self._process_request('GetQualificationType', params, [('QualificationType', QualificationType)])