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
def search_qualification_types(self, query=None, sort_by='Name', sort_direction='Ascending', page_size=10, page_number=1, must_be_requestable=True, must_be_owned_by_caller=True):
    """TODO: Document."""
    params = {'Query': query, 'SortProperty': sort_by, 'SortDirection': sort_direction, 'PageSize': page_size, 'PageNumber': page_number, 'MustBeRequestable': must_be_requestable, 'MustBeOwnedByCaller': must_be_owned_by_caller}
    return self._process_request('SearchQualificationTypes', params, [('QualificationType', QualificationType)])