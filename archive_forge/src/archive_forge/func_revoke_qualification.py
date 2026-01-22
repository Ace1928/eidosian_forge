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
def revoke_qualification(self, subject_id, qualification_type_id, reason=None):
    """TODO: Document."""
    params = {'SubjectId': subject_id, 'QualificationTypeId': qualification_type_id, 'Reason': reason}
    return self._process_request('RevokeQualification', params)