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
def change_hit_type_of_hit(self, hit_id, hit_type):
    """
        Change the HIT type of an existing HIT. Note that the reward associated
        with the new HIT type must match the reward of the current HIT type in
        order for the operation to be valid.

        :type hit_id: str
        :type hit_type: str
        """
    params = {'HITId': hit_id, 'HITTypeId': hit_type}
    return self._process_request('ChangeHITTypeOfHIT', params)