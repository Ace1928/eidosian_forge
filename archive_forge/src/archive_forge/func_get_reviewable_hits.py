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
def get_reviewable_hits(self, hit_type=None, status='Reviewable', sort_by='Expiration', sort_direction='Ascending', page_size=10, page_number=1):
    """
        Retrieve the HITs that have a status of Reviewable, or HITs that
        have a status of Reviewing, and that belong to the Requester
        calling the operation.
        """
    params = {'Status': status, 'SortProperty': sort_by, 'SortDirection': sort_direction, 'PageSize': page_size, 'PageNumber': page_number}
    if hit_type is not None:
        params.update({'HITTypeId': hit_type})
    return self._process_request('GetReviewableHITs', params, [('HIT', HIT)])