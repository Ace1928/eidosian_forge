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
@staticmethod
def get_price_as_price(reward):
    """
        Returns a Price data structure from either a float or a Price
        """
    if isinstance(reward, Price):
        final_price = reward
    else:
        final_price = Price(reward)
    return final_price