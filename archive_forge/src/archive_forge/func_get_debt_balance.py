import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@requires(['CreditInstrumentId'])
@api_action()
def get_debt_balance(self, action, response, **kw):
    """
        Returns the balance corresponding to the given credit instrument.
        """
    return self.get_object(action, kw, response)