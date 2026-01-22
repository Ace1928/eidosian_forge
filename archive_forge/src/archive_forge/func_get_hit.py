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
def get_hit(self, hit_id, response_groups=None):
    """
        """
    params = {'HITId': hit_id}
    if response_groups:
        self.build_list_params(params, response_groups, 'ResponseGroup')
    return self._process_request('GetHIT', params, [('HIT', HIT)])